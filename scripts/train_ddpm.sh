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

# AFHQ training
# python main/train_ddpm.py +dataset=afhq128/train \
#                      dataset.ddpm.data.root='/data1/kushagrap20/reconstructions/afhq_reconsv2/' \
#                      dataset.ddpm.data.name='recons' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.batch_size=12 \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/ddpm_afhq_16thDec_form1_scale[-11]\' \
#                      dataset.ddpm.training.restore_path=\'/data1/kushagrap20/ddpm_afhq_13thDec_form1_scale[-11]/checkpoints/ddpmv2-afhq_13thDec_form1_scale[-11]-epoch=402-loss=0.0045.ckpt\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix=\'afhq_16thDec_form1_scale[-11]\'

# CelebA training
# python main/train_ddpm.py +dataset=celeba64/train \
#                      dataset.ddpm.data.root='/data1/kushagrap20/vae_celeba64_recons' \
#                      dataset.ddpm.data.name='recons' \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.data.norm='True' \
#                      dataset.ddpm.training.type='form2' \
#                      dataset.ddpm.training.batch_size=32 \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/ddpm_celeba64_4thDec_form2_sota\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix='celebamaskhq_celeba64_4thDec_form2_sota'

# CIFAR-10 training
python main/train_ddpm.py +dataset=cifar10/train \
                     dataset.ddpm.data.root='/data1/kushagrap20/reconstructions/cifar10/' \
                     dataset.ddpm.data.name='recons' \
                     dataset.ddpm.data.norm=True \
                     dataset.ddpm.data.hflip=True \
                     dataset.ddpm.training.type='form1' \
                     dataset.ddpm.training.epochs=900 \
                     dataset.ddpm.training.batch_size=64 \
                     dataset.ddpm.training.device=\'gpu:2,3\' \
                     dataset.ddpm.training.results_dir=\'/data1/kushagrap20/ddpm_cifar10_form1_scale=[-1,1]_28thDec_wavegrad_nc/\' \
                     dataset.ddpm.training.restore_path=\'/data1/kushagrap20/ddpm_cifar10_form1_scale=[-1,1]_28thDec_wavegrad_nc/checkpoints/ddpmv2-cifar10_form1_scale=[-1,1]_28thDec_wavegrad_nc-epoch=735-loss=0.0114.ckpt\' \
                     dataset.ddpm.training.workers=4 \
                     dataset.ddpm.training.chkpt_prefix=\'cifar10_form1_scale=[-1,1]_28thDec_wavegrad_nc\'