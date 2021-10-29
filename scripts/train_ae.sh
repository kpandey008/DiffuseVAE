# CelebAMaskHQ training
python main/train_ae.py +dataset=celebamaskhq128/train \
                     dataset.vae.data.root='/data/kushagrap20/datasets/CelebAMask-HQ/' \
                     dataset.vae.data.name='celebamaskhq' \
                     dataset.vae.training.batch_size=32 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:0,1,2,3\' \
                     dataset.vae.training.results_dir=\'/data/kushagrap20/vae_alpha_0_1/\' \
                     dataset.vae.training.restore_path=\'/data/kushagrap20/vae_alpha_0_1/checkpoints/vae-celebamaskhq_alpha_0_1-epoch=999-train_loss=0.0000.ckpt\' \
                     dataset.vae.training.workers=4 \
                     dataset.vae.training.chkpt_prefix='celebamaskhq_alpha_0_1' \
                     dataset.vae.training.alpha=0.1