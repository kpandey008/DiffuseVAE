# CelebAMaskHQ training
# python main/train_ae.py +dataset=celebamaskhq128/train \
#                      dataset.vae.data.root='/data/kushagrap20/datasets/CelebAMask-HQ/' \
#                      dataset.vae.data.name='celebamaskhq' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data/kushagrap20/vae_alpha_0_1/\' \
#                      dataset.vae.training.restore_path=\'/data/kushagrap20/vae_alpha_0_1/checkpoints/vae-celebamaskhq_alpha_0_1-epoch=999-train_loss=0.0000.ckpt\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix='celebamaskhq_alpha_0_1' \
#                      dataset.vae.training.alpha=0.1

# AFHQ training
# python main/train_ae.py +dataset=afhq128/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/afhq/' \
#                      dataset.vae.data.name='afhq' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_afhq_alpha=1.0/\' \
#                      dataset.vae.training.restore_path=\'/data1/kushagrap20/vae_afhq_alpha=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1343-train_loss=0.0000.ckpt\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0


# CelebA training
# python main/train_ae.py +dataset=celeba64/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
#                      dataset.vae.data.name='celeba' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_celeba64_alpha=1.0_try/\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0