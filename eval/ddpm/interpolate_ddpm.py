import copy
import os

import hydra
import torch
from models.diffusion import DDPM, DDPMWrapper, SuperResModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from tqdm import tqdm
from util import plot_interpolations, configure_device, save_as_images


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="configs")
def interpolate_ddpm(config):
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    dev, _ = configure_device(config_ddpm.evaluation.device)
    image_size = config_ddpm.data.image_size
    z_dim = config_ddpm.model.z_dim
    n_steps = config_ddpm.evaluation.n_steps

    # Lambdas for interpolation
    lam = torch.linspace(0, 1.0, steps=config_ddpm.interpolation.n_steps, device=dev)

    # VAE model
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
        enc_block_str=config_vae.model.enc_block_config,
        dec_block_str=config_vae.model.dec_block_config,
        enc_channel_str=config_vae.model.enc_channel_config,
        dec_channel_str=config_vae.model.dec_channel_config,
    )
    vae.eval()

    # Superres Model
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,
        model_channels=config_ddpm.model.dim,
        out_channels=3,
        num_res_blocks=config_ddpm.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,
        num_heads=config_ddpm.model.n_heads,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    online_ddpm = DDPM(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
    )
    target_ddpm = DDPM(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
    )
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        conditional=True,
        strict=False,
        pred_steps=n_steps,
        eval_mode="recons",
    )

    ddpm_wrapper.to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []

    with torch.no_grad():
        # Interpolate in the DDPM latent space
        z_1 = torch.randn(1, z_dim, 1, 1, device=dev)
        recons_inter = vae(z_1)

        if config_ddpm.data.norm:
            recons_inter = 2 * recons_inter - 1

        x_t1 = config_ddpm.evaluation.temp * torch.randn(
            1, 3, image_size, image_size, device=dev
        )
        x_t2 = config_ddpm.evaluation.temp * torch.randn(
            1, 3, image_size, image_size, device=dev
        )

        for idx, l in tqdm(enumerate(lam)):
            # Sample from DDPM
            x_t_inter = x_t1 * l + x_t2 * (1 - l)
            ddpm_sample = ddpm_wrapper(x_t_inter, cond=recons_inter, n_steps=n_steps)[
                str(n_steps)
            ].cpu()
            ddpm_samples_list.append(ddpm_sample)
            vae_samples_list.append(recons_inter.cpu())

    cat_ddpm_samples = torch.cat(ddpm_samples_list, dim=0)
    cat_vae_samples = torch.cat(vae_samples_list, dim=0)

    # Save DDPM and VAE samples
    save_path = config_ddpm.evaluation.save_path
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    save_as_images(cat_ddpm_samples, file_name=os.path.join(save_path, "inter_ddpm"))
    save_as_images(cat_vae_samples, file_name=os.path.join(save_path, "inter_vae"))

    # Compare
    save_path = config_ddpm.evaluation.save_path
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    plot_interpolations(
        ddpm_samples_list, save_path=os.path.join(save_path, "inter_plot.png")
    )


if __name__ == "__main__":
    interpolate_ddpm()
