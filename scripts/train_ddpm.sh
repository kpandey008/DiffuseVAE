# # CIFAR-10 (Form-1)
# python main/train_ddpm.py +dataset=cifar10/train \
#                      dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/\' \
#                      dataset.ddpm.data.name='cifar10' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.3 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=2850 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=32 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1_28thJuly_sota_nheads=8_dropout=0.3/\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'cifar10_rework_form1_28thJuly_sota_nheads=8_dropout=0.3\'


# # CelebA-64 (Form-1)
# python main/train_ddpm.py +dataset=celeba64/train \
#                      dataset.ddpm.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
#                      dataset.ddpm.data.name='celeba' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,2,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=500 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=32 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/celeba64/vae_celeba64_alpha=1.0/checkpoints/vae-celeba64_alpha=1.0-epoch=245-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1/\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1\'


# # CelebAHQ-128 (Form1)
# python train_ddpm.py +dataset=celebamaskhq128/train \
#                      dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/CelebAMask-HQ\' \
#                      dataset.ddpm.data.name='celebamaskhq' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=8 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/res_128/vae-cmhq128_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_celebahq128_rework_form1__21stJune_sota_nheads=8_dropout=0.1/\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'celebahq128_rework_form1__21stJune_sota_nheads=8_dropout=0.1\'


# # CelebAHQ-256 (Form1)
# python train_ddpm.py +dataset=celebahq/train \
#                      dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/celeba_hq/\' \
#                      dataset.ddpm.data.name='celebahq' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,1,2,2,4,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=8 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/vae-afhq256_10thJuly_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'tpu\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_chq256_rework_form1_10thJuly_sota_nheads=8_dropout=0.1/\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'chq256_rework_form1_10thJuly_sota_nheads=8_dropout=0.1\'