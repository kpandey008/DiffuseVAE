from models.diffusion.unet_openai import checkpoint
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T


class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae=None,
        lr=2e-5,
        n_anneal_steps=0,
        loss="l1",
        sample_from="target",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        self.sample_from = sample_from
        self.conditional = conditional
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae

        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.n_anneal_steps = n_anneal_steps
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints

        # Disable automatic optimization
        self.automatic_optimization = False

    def forward(self, x, cond=None, n_steps=None, checkpoints=[]):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        return sample_nw.sample(x, cond=cond, n_steps=n_steps, checkpoints=checkpoints)

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        if self.conditional:
            cond, x = batch
        else:
            x = batch

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(x, eps, t, low_res=cond)

        # Compute loss
        loss = self.criterion(eps, eps_pred)
        self.log("loss", loss)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.online_network.decoder.parameters(), 1.0)
        optim.step()

        # Scheduler step
        lr_sched.step()
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            x_t = batch
            return self(
                x_t,
                cond=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            # This will be broadcasted automatically
            recons = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(recons)
        else:
            (recons, _), x_t = batch
            x_t = x_t[0]  # This is really a one element tuple
        return (
            self(
                x_t,
                cond=recons,
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
