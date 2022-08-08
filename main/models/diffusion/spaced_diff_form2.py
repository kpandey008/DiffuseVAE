# CREDITS: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/respace.py
import torch.nn as nn
import torch


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SpacedDiffusionForm2(nn.Module):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    """

    def __init__(
        self,
        base_diffusion,
        use_timesteps,
    ):
        super().__init__()
        self.base_diffusion = base_diffusion
        self.use_timesteps = use_timesteps
        self.timestep_map = []
        self.original_num_steps = self.base_diffusion.T
        self.decoder = self.base_diffusion.decoder
        self.var_type = self.base_diffusion.var_type

        last_alpha_cumprod = 1.0
        alphas_cumprod = torch.cumprod(1.0 - self.base_diffusion.betas, dim=0)
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(torch.tensor([1 - alpha_cumprod / last_alpha_cumprod]))
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.register_buffer("betas", torch.cat(new_betas))
        dev = self.betas.device
        alphas = 1.0 - self.betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))
        self.register_buffer(
            "alpha_bar_shifted",
            torch.cat([torch.tensor([1.0], device=dev), self.alpha_bar[:-1]]),
        )

        assert self.alpha_bar_shifted.shape == torch.Size(
            [
                len(self.timestep_map),
            ]
        )

        # Auxillary consts
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("minus_sqrt_alpha_bar", torch.sqrt(1.0 - self.alpha_bar))
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alpha_bar)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alpha_bar - 1)
        )

        # Posterior q(x_t-1|x_t,x_0,t) covariance of the forward process
        self.register_buffer(
            "post_variance",
            self.betas * (1.0 - self.alpha_bar_shifted) / (1.0 - self.alpha_bar),
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
            self.betas * torch.sqrt(self.alpha_bar_shifted) / (1.0 - self.alpha_bar),
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(alphas) * (1 - self.alpha_bar_shifted) / (1 - self.alpha_bar),
        )
        self.register_buffer(
            "post_coeff_3",
            1 - self.post_coeff_2,
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

    def get_posterior_mean_covariance(
        self, x_t, t, clip_denoised=True, cond=None, z_vae=None, guidance_weight=0.0
    ):
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        t_model_ = torch.full(
            (x_t.size(0),), self.timestep_map[t], device=x_t.device, dtype=torch.long
        )
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )
        x_hat = 0 if cond is None else cond

        if guidance_weight == 0:
            eps = self.decoder(x_t, t_model_, low_res=cond, z=z_vae)
        else:
            eps = (1 + guidance_weight) * self.decoder(
                x_t, t_model_, low_res=cond, z=z_vae
            ) - guidance_weight * self.decoder(
                x_t,
                t_model_,
                low_res=torch.zeros_like(cond),
                z=torch.zeros_like(z_vae) if z_vae is not None else None,
            )

        # Generate the reconstruction from x_t
        x_recons = self._predict_xstart_from_eps(x_t, t_, eps, cond=cond)

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
        return post_mean, post_variance, post_log_variance, x_recons, eps

    def forward(
        self,
        x_t,
        cond=None,
        z_vae=None,
        guidance_weight=0.0,
        checkpoints=[],
        ddpm_latents=None,
    ):
        # The sampling process goes here!
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}

        if ddpm_latents is not None:
            ddpm_latents = ddpm_latents.to(x_t.device)

        num_steps = len(self.timestep_map)
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
                _,
                _,
            ) = self.get_posterior_mean_covariance(
                x,
                t,
                cond=cond,
                z_vae=z_vae,
                guidance_weight=guidance_weight,
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

    def get_ddim_mean_cov(
        self,
        x,
        t,
        clip_denoised=True,
        cond=None,
        z_vae=None,
        eta=0.0,
        guidance_weight=0.0,
    ):
        B = x.size(0)
        t_ = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        t_model_ = torch.full(
            (x.size(0),), self.timestep_map[t], device=x.device, dtype=torch.long
        )
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )

        # Generate the reconstruction from x_t
        if guidance_weight == 0:
            eps = self.decoder(x, t_model_, low_res=cond, z=z_vae)
        else:
            eps = (1 + guidance_weight) * self.decoder(
                x, t_model_, low_res=cond, z=z_vae
            ) - guidance_weight * self.decoder(
                x,
                t_model_,
                low_res=torch.zeros_like(cond),
                z=torch.zeros_like(z_vae) if z_vae is not None else None,
            )
        x_recons = self._predict_xstart_from_eps(x, t_, eps, cond=cond)

        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)

        alpha_bar = extract(self.alpha_bar, t_, x.shape)
        alpha_bar_prev = extract(self.alpha_bar_shifted, t_, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        coeff = 1 - torch.sqrt(1 - alpha_bar_prev - sigma ** 2) / torch.sqrt(
            1 - alpha_bar
        )

        # Compute mean
        x_hat = 0 if cond is None else cond
        mean_pred = (
            x_recons * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2)
            * (eps + x_hat / torch.sqrt(1 - alpha_bar))
            + coeff * x_hat
        )
        return mean_pred, sigma

    def ddim_sample(
        self, x_t, cond=None, z_vae=None, checkpoints=[], eta=0.0, guidance_weight=0.0
    ):
        # The sampling process goes here!
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}

        num_steps = len(self.timestep_map)
        checkpoints = [num_steps] if checkpoints == [] else checkpoints
        for idx, t in enumerate(reversed(range(0, num_steps))):
            z = torch.randn_like(x_t)
            assert z.shape == x_t.shape
            (post_mean, post_variance,) = self.get_ddim_mean_cov(
                x,
                t,
                cond=cond,
                z_vae=z_vae,
                eta=eta,
                guidance_weight=guidance_weight,
            )
            nonzero_mask = (
                torch.tensor(t != 0, device=x.device)
                .float()
                .view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0

            # Langevin step!
            x = post_mean + nonzero_mask * post_variance * z

            if t == 0:
                # NOTE: In the final step we remove the vae reconstruction bias
                # added to the images as it degrades quality
                x -= cond

            # Add results
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x
        return sample_dict
