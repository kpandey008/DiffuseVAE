import torch
import torch.nn as nn


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(nn.Module):
    def __init__(
        self,
        decoder,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
        var_type="fixedlarge",
    ):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.var_type = var_type

        # Main constants
        self.register_buffer(
            "betas", torch.linspace(self.beta_1, self.beta_2, steps=self.T).double()
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
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("minus_sqrt_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_bar - 1)
        )

        # Posterior q(x_t-1|x_t,x_0,t) covariance of the forward process
        self.register_buffer(
            "post_variance", self.betas * (1.0 - alpha_bar_shifted) / (1.0 - alpha_bar)
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
        )

        # q(x_t-1 | x_t, x_0) mean coefficients
        self.register_buffer(
            "post_coeff_1",
            self.betas * torch.sqrt(alpha_bar_shifted) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(alphas) * (1 - alpha_bar_shifted) / (1 - alpha_bar),
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def get_posterior_mean_covariance(
        self, x_t, t, clip_denoised=True, cond=None, z_vae=None, guidance_weight=0.0
    ):
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )

        # Compute updated score
        if guidance_weight == 0:
            eps_score = self.decoder(x_t, t_, low_res=cond, z=z_vae)
        else:
            eps_score = (1 + guidance_weight) * self.decoder(
                x_t, t_, low_res=cond, z=z_vae
            ) - guidance_weight * self.decoder(
                x_t,
                t_,
                low_res=torch.zeros_like(cond),
                z=torch.zeros_like(z_vae) if z_vae is not None else None,
            )

        # Generate the reconstruction from x_t
        x_recons = self._predict_xstart_from_eps(x_t, t_, eps_score)

        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)

        # Compute posterior mean from the reconstruction
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
        )

        # Extract posterior variance
        p_variance, p_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            "fixedlarge": (
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=x_t.device),
                        self.betas[1:],
                    ]
                ),
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

    def sample(
        self,
        x_t,
        cond=None,
        z_vae=None,
        n_steps=None,
        guidance_weight=0.0,
        checkpoints=[],
        ddpm_latents=None,
    ):
        # The sampling process goes here. This sampler also supports truncated sampling.
        # For spaced sampling (used in DDIM etc.) see SpacedDiffusion model in spaced_diff.py
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}

        if ddpm_latents is not None:
            ddpm_latents = ddpm_latents.to(x_t.device)

        num_steps = self.T if n_steps is None else n_steps
        checkpoints = [num_steps] if checkpoints == [] else checkpoints

        for idx, t in enumerate(reversed(range(0, num_steps))):
            z = (
                torch.randn_like(x_t)
                if ddpm_latents is None
                else torch.stack([ddpm_latents[idx]] * B)
            )
            assert z.shape == x_t.shape
            (
                post_mean,
                post_variance,
                post_log_variance,
            ) = self.get_posterior_mean_covariance(
                x, t, cond=cond, z_vae=z_vae, guidance_weight=guidance_weight
            )
            nonzero_mask = (
                torch.tensor(t != 0, device=x.device)
                .float()
                .view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0

            # Langevin step!
            x = post_mean + nonzero_mask * torch.exp(0.5 * post_log_variance) * z

            # Add results
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x
        return sample_dict

    def compute_noisy_input(self, x_start, eps, t):
        assert eps.shape == x_start.shape
        # Samples the noisy input x_t ~ q(x_t|x_0) in the forward process
        return x_start * extract(self.sqrt_alpha_bar, t, x_start.shape) + eps * extract(
            self.minus_sqrt_alpha_bar, t, x_start.shape
        )

    def forward(self, x, eps, t, low_res=None, z=None):
        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t)
        return self.decoder(x_t, t, low_res=low_res, z=z)
