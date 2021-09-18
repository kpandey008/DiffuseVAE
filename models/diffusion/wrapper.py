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
        vae,
        lr=2e-5,
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
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints

    def forward(self, x, cond=None, n_steps=None, checkpoints=[]):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        return sample_nw.sample(x, cond=cond, n_steps=n_steps, checkpoints=checkpoints)

    def training_step(self, batch, batch_idx):
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
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
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
        optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        return optimizer
