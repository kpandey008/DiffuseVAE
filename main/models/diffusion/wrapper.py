import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.diffusion.ddpm_form2 import DDPMv2


class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        z_cond=False,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        self.sample_from = sample_from
        self.conditional = conditional
        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp

        # Disable automatic optimization
        self.automatic_optimization = False

    def forward(self, x, cond=None, z=None, n_steps=None, checkpoints=[]):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        return sample_nw.sample(x, cond=cond, z_vae=z, n_steps=n_steps, checkpoints=checkpoints)

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        if self.conditional:
            x = batch
            with torch.no_grad():
                mu, logvar = self.vae.encode(x * 0.5 + 0.5)
                z = self.vae.reparameterize(mu, logvar)
                cond = self.vae.decode(z)
                cond = 2 * cond - 1
        else:
            x = batch

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(x, eps, t, low_res=cond, z=z if self.z_cond else None)

        # Compute loss
        loss = self.criterion(eps, eps_pred)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)

            # Initial temperature scaling
            x_t = x_t * self.temp
        else:
            (recons, _), x_t = batch
            x_t = self.temp * x_t[0]  # This is really a one element tuple

        # Normalize
        recons = 2 * recons - 1

        # Formulation-2 initial latent
        if isinstance(self.online_network, DDPMv2):
            x_t = recons + self.temp * torch.randn_like(recons)

        return (
            self(
                x_t,
                cond=recons,
                z=z if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
            ),
            recons,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
