# CelebAMaskHQ training
# python train_ddpm.py +dataset=celebamaskhq128/train \
#                      dataset.ddpm.data.root='/data/kushagrap20/vaedm/reconstructions_celebahq' \
#                      dataset.ddpm.data.name='recons' \
#                      dataset.ddpm.data.norm='False' \
#                      dataset.ddpm.training.type='form2' \
#                      dataset.ddpm.training.batch_size=10 \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data/kushagrap20/ddpm_celebamaskhq_26thOct_form2_scale[01]\' \
#                      dataset.ddpm.training.restore_path=\'/data/kushagrap20/ddpm_celebamaskhq_26thOct_form2_scale[01]/checkpoints/ddpmv2-celebamaskhq_26thOct_form2_scale01-epoch=07-loss=0.0017.ckpt\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix='celebamaskhq_26thOct_form2_scale01'

# CelebA trainig
python main/train_ddpm.py +dataset=celeba64/train \
                     dataset.ddpm.data.root='/data1/kushagrap20/vae_celeba64_recons' \
                     dataset.ddpm.data.name='recons' \
                     dataset.ddpm.data.hflip=True \
                     dataset.ddpm.data.norm='True' \
                     dataset.ddpm.training.type='form2' \
                     dataset.ddpm.training.batch_size=32 \
                     dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
                     dataset.ddpm.training.results_dir=\'/data1/kushagrap20/ddpm_celeba64_21stNov_form1_sota\' \
                     dataset.ddpm.training.restore_path=\'/data1/kushagrap20/ddpm_celeba64_21stNov_form1_sota/checkpoints/ddpmv2-celebamaskhq_celeba64_21stNov_form1_sota-epoch=499-loss=0.0081.ckptd\' \
                     dataset.ddpm.training.workers=2 \
                     dataset.ddpm.training.chkpt_prefix='celebamaskhq_celeba64_21stNov_form1_sota'