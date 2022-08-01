# # CelebA-64 training
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
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1/\' \
#                      dataset.ddpm.training.restore_path=\'/data1/kushagrap20/diffusevae_celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1/checkpoints/ddpmv2-celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1-epoch=427-loss=0.0145.ckpt\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1\'

# CIFAR-10 training
python main/train_ddpm.py +dataset=cifar10/train \
                     dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/\' \
                     dataset.ddpm.data.name='cifar10' \
                     dataset.ddpm.data.norm=True \
                     dataset.ddpm.data.hflip=True \
                     dataset.ddpm.model.dim=160 \
                     dataset.ddpm.model.dropout=0.3 \
                     dataset.ddpm.model.attn_resolutions=\'16,\' \
                     dataset.ddpm.model.n_residual=3 \
                     dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                     dataset.ddpm.model.n_heads=8 \
                     dataset.ddpm.training.type='form1' \
                     dataset.ddpm.training.cfd_rate=0.0 \
                     dataset.ddpm.training.epochs=2850 \
                     dataset.ddpm.training.z_cond=False \
                     dataset.ddpm.training.batch_size=32 \
                     dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
                     dataset.ddpm.training.device=\'gpu:0\' \
                     dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1_28thJuly_sota_nheads=8_dropout=0.3_largermodel/\' \
                     dataset.ddpm.training.workers=2 \
                     dataset.ddpm.training.chkpt_prefix=\'cifar10_rework_form1_28thJuly_sota_nheads=8_dropout=0.3_largermodel\'

# # CIFAR-10 training (with conditional dropout)
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
#                      dataset.ddpm.training.cfd_rate=0.1 \
#                      dataset.ddpm.training.epochs=2850 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=32 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance/\' \
#                      dataset.ddpm.training.restore_path=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance/checkpoints/ddpmv2-cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance-epoch=2450-loss=0.0219.ckpt\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix=\'cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance\'