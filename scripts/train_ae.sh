# # CelebAMaskHQ training
# python main/train_ae.py +dataset=celebamaskhq128/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/CelebAMask-HQ/' \
#                      dataset.vae.data.name='celebamaskhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=42 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_cmhq128_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'cmhq128_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # FFHQ 128 training
# python main/train_ae.py +dataset=ffhq/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/ffhq/' \
#                      dataset.vae.data.name='ffhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_ffhq128_11thJune_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'ffhq128_11thJune_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # AFHQv2 training
# python main/train_ae.py +dataset=afhq256/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/afhq_v2/' \
#                      dataset.vae.data.name='afhq' \
#                      dataset.vae.training.batch_size=8 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_afhq256_10thJuly_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq256_10thJuly_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0


# # CelebA training
# python main/train_ae.py +dataset=celeba64/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
#                      dataset.vae.data.name='celeba' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_celeba64_alpha=1.0/\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0