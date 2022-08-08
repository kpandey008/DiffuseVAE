# Add project directory to sys.path
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import copy
import os

import hydra
import torch
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from tqdm import tqdm
from util import compare_interpolations, configure_device, save_as_images
from joblib import load


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path=os.path.join(p, "configs"))
def interpolate_vae(config):
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    # HARDCODED
    dev = "cuda:0"
    image_size = config_ddpm.data.image_size
    z_dim = config_vae.model.z_dim
    n_steps = config_ddpm.evaluation.n_steps
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Lambdas for interpolation
    lam = torch.linspace(0, 1.0, steps=config_ddpm.interpolation.n_steps, device=dev)

    # VAE model
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
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
        z_dim=config_ddpm.evaluation.z_dim,
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,
        use_z=config_ddpm.evaluation.z_cond,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="sample",
        resample_strategy=config_ddpm.evaluation.resample_strategy,
        skip_strategy=config_ddpm.evaluation.skip_strategy,
        sample_method=config_ddpm.evaluation.sample_method,
        sample_from=config_ddpm.evaluation.sample_from,
        data_norm=config_ddpm.data.norm,
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,
        ddpm_latents=ddpm_latents,
        strict=True,
    )

    ddpm_wrapper.to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []
    expde_model_path = config_vae.evaluation.expde_model_path

    with torch.no_grad():
        # Interpolate in the VAE latent space
        z_1 = torch.randn(1, z_dim, 1, 1, device=dev).float()
        z_2 = torch.randn(1, z_dim, 1, 1, device=dev).float()

        if expde_model_path is not None and expde_model_path != "":
            print(
                "Found an Ex-PDE model. Will sample latents for interpolation from it instead!"
            )
            gmm = load(expde_model_path)
            gmm.set_params(random_state=config_ddpm.evaluation.seed)
            z = gmm.sample(2)[0]
            z_1 = torch.from_numpy(z[0]).view(1, z_dim, 1, 1).to(dev).float()
            z_2 = torch.from_numpy(z[1]).view(1, z_dim, 1, 1).to(dev).float()
            assert z_1.size() == (1, z_dim, 1, 1)

        x_t = None

        for _, l in tqdm(enumerate(lam)):
            # Sample from VAE
            z_inter = z_1 * l + z_2 * (1 - l)
            recons_inter = vae(z_inter)
            vae_samples_list.append(recons_inter.cpu())

            # Sample from DDPM
            if config_ddpm.data.norm:
                recons_inter = 2 * recons_inter - 1

            if x_t is None:
                x_t = (config_ddpm.evaluation.temp * torch.randn_like(recons_inter)).to(
                    dev
                )
                if config_ddpm.evaluation.type == "form2":
                    x_t = recons_inter + x_t

            ddpm_sample = ddpm_wrapper(
                x_t,
                cond=recons_inter,
                z=z_inter if config_ddpm.evaluation.z_cond is True else None,
                n_steps=n_steps,
                ddpm_latents=ddpm_latents,
            )[str(n_steps)].cpu()
            ddpm_samples_list.append(ddpm_sample)

    cat_ddpm_samples = torch.cat(ddpm_samples_list, dim=0)
    cat_vae_samples = torch.cat(vae_samples_list, dim=0)

    # Save DDPM and VAE samples
    save_path = config_ddpm.evaluation.save_path
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    save_as_images(
        cat_ddpm_samples,
        file_name=os.path.join(save_path, "inter_ddpm"),
        denorm=config_ddpm.data.norm,
    )
    save_as_images(
        cat_vae_samples,
        file_name=os.path.join(save_path, "inter_vae"),
        denorm=False,
    )

    # # Compare
    save_path = config_ddpm.evaluation.save_path
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    compare_interpolations(
        ddpm_samples_list,
        vae_samples_list,
        save_path=os.path.join(save_path, "inter_compare.png"),
        denorm=config_ddpm.data.norm,
    )


if __name__ == "__main__":
    interpolate_vae()
