import torch
import torch.nn as nn

from models.diffusion.unet_openai import UNetModel


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPMv2(nn.Module):
    def __init__(
        self,
        decoder,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
        var_type="fixedlarge",
        ddpm_latents=None,
        persistent_buffers=True,
    ):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.var_type = var_type
        self.ddpm_latents = ddpm_latents
        self.persistent_buffers = persistent_buffers

        # Main constants
        self.register_buffer(
            "betas",
            torch.linspace(self.beta_1, self.beta_2, steps=self.T).double(),
            persistent=self.persistent_buffers,
        )
        dev = self.betas.device
        alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_shifted = torch.cat([torch.tensor([1.0], device=dev), alpha_bar[:-1]])

        assert alpha_bar_shifted.shape == torch.Size(
            [
                self.T,
            ]
        )

        # Auxillary consts
        self.register_buffer(
            "sqrt_alpha_bar",
            torch.sqrt(alpha_bar),
            persistent=self.persistent_buffers,
        )
        self.register_buffer(
            "sqrt_alpha_bar_shifted",
            torch.sqrt(alpha_bar_shifted),
            persistent=self.persistent_buffers,
        )
        self.register_buffer(
            "minus_sqrt_alpha_bar",
            torch.sqrt(1.0 - alpha_bar),
            persistent=self.persistent_buffers,
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alpha_bar),
            persistent=self.persistent_buffers,
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alpha_bar - 1),
            persistent=self.persistent_buffers,
        )

        # Posterior q(x_t-1|x_t,x_0,t) covariance of the forward process
        self.register_buffer(
            "post_variance",
            self.betas * (1.0 - alpha_bar_shifted) / (1.0 - alpha_bar),
            persistent=self.persistent_buffers,
        )
        # Clipping because post_variance is 0 before the chain starts
        self.register_buffer(
            "post_log_variance_clipped",
            torch.log(
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=dev),
                        self.post_variance[1:],
                    ]
                )
            ),
            persistent=self.persistent_buffers,
        )

        # q(x_t-1 | x_t, x_0) mean coefficients
        self.register_buffer(
            "post_coeff_1",
            self.betas * torch.sqrt(alpha_bar_shifted) / (1.0 - alpha_bar),
            persistent=self.persistent_buffers,
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(alphas) * (1 - alpha_bar_shifted) / (1 - alpha_bar),
            persistent=self.persistent_buffers,
        )
        self.register_buffer(
            "post_coeff_3",
            1 - self.post_coeff_2,
            persistent=self.persistent_buffers,
        )

    def _predict_xstart_from_eps(self, x_t, t, eps, cond=None):
        assert x_t.shape == eps.shape
        x_hat = 0 if cond is None else cond
        assert x_hat.shape == x_t.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def get_posterior_mean_covariance(self, x_t, t, clip_denoised=True, cond=None):
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        sqrt_alpha_bar_ = torch.full(
            (x_t.size(0),), self.sqrt_alpha_bar[t], device=x_t.device, dtype=torch.float
        )
        assert sqrt_alpha_bar_.shape == torch.Size(
            [
                B,
            ]
        )

        assert t_.shape == torch.Size(
            [
                B,
            ]
        )
        x_hat = 0 if cond is None else cond

        # Generate the reconstruction from x_t
        x_recons = self._predict_xstart_from_eps(
            x_t, t_, self.decoder(x_t, sqrt_alpha_bar_, low_res=cond), cond=cond
        )

        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)

        # Compute posterior mean from the reconstruction
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
            + extract(self.post_coeff_3, t_, x_t.shape) * x_hat
        )

        # Extract posterior variance
        p_variance, p_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            "fixedlarge": (
                self.betas,
                torch.log(
                    torch.cat(
                        [
                            torch.tensor([self.post_variance[1]], device=x_t.device),
                            self.betas[1:],
                        ]
                    )
                ),
            ),
            "fixedsmall": (
                self.post_variance,
                self.post_log_variance_clipped,
            ),
        }[self.var_type]
        post_variance = extract(p_variance, t_, x_t.shape)
        post_log_variance = extract(p_log_variance, t_, x_t.shape)
        return post_mean, post_variance, post_log_variance

    def sample(self, x_t, cond=None, n_steps=None, checkpoints=[]):
        # The sampling process goes here!
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}

        if self.ddpm_latents is not None:
            self.ddpm_latents = self.ddpm_latents.to(x_t.device)

        num_steps = self.T if n_steps is None else n_steps
        checkpoints = [num_steps] if checkpoints == [] else checkpoints
        for idx, t in enumerate(reversed(range(0, num_steps))):
            z = (
                torch.randn_like(x_t)
                if self.ddpm_latents is None
                else torch.stack([self.ddpm_latents[idx]] * B)
            )
            (
                post_mean,
                post_variance,
                post_log_variance,
            ) = self.get_posterior_mean_covariance(
                x,
                t,
                cond=cond,
            )
            nonzero_mask = (
                torch.tensor(t != 0, device=x.device)
                .float()
                .view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0

            # Langevin step!
            x = post_mean + nonzero_mask * torch.exp(0.5 * post_log_variance) * z

            if t == 0:
                # NOTE: In the final step we remove the vae reconstruction bias
                # added to the images as it degrades quality
                x -= cond

            # Add results
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x
        return sample_dict

    def compute_noisy_input(self, x_start, eps, sqrt_alphas, low_res=None):
        assert eps.shape == x_start.shape
        x_hat = 0 if low_res is None else low_res
        # Samples the noisy input x_t ~ N(x_t|x_0) in the forward process
        B, *_ = sqrt_alphas.shape
        sqrt_alphas_ = sqrt_alphas.reshape(B, *((1,) * (len(x_start.shape) - 1)))
        minus_sqrt_alphas_ = torch.sqrt(1 - torch.pow(sqrt_alphas_, 2))
        return x_start * sqrt_alphas_ + x_hat + eps * minus_sqrt_alphas_

    def forward(self, x, eps, t, low_res=None):
        # Sample continuous noise
        sqrt_alphas = torch.distributions.uniform.Uniform(
            low=self.sqrt_alpha_bar[t], high=self.sqrt_alpha_bar_shifted[t]
        ).sample(sample_shape=(1,))[0]

        # Predict noise
        x_t = self.compute_noisy_input(x, eps, sqrt_alphas, low_res=low_res)
        return self.decoder(x_t, sqrt_alphas, low_res=low_res)


if __name__ == "__main__":
    decoder = UNetModel(
        3,
        64,
        3,
        2,
        [
            16,
        ],
    )
    ddpm = DDPMv2(decoder)
    t = torch.randint(0, 1000, size=(4,))
    sample = torch.randn(4, 3, 128, 128)
    loss = ddpm(sample, torch.randn_like(sample), t)
    print(loss)

    # Test sampling
    x_t = torch.randn(4, 3, 128, 128)
    samples = ddpm.sample(x_t)
    print(samples.shape)
