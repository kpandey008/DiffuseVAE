python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.data.ddpm_latent_path=\'/data1/kushagrap20/cmhq_ddpm_latents_10.pt\' \
                        dataset.ddpm.model.beta1=1e-6 \
                        dataset.ddpm.model.beta2=0.6 \
                        dataset.ddpm.model.n_timesteps=10 \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_1' \
                        dataset.ddpm.evaluation.device=\'gpu:0,1\' \
                        dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_4thNov_form2_scale[-11]_wavegrad_nc-epoch=251-loss=0.0158.ckpt\' \
                        dataset.ddpm.evaluation.type='form2' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=16 \
                        dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_form1_cont_samples_fixedddpm_25thDec/\' \
                        dataset.ddpm.evaluation.n_samples=512 \
                        dataset.ddpm.evaluation.n_steps=10 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.ddpm.evaluation.persistent_buffers=False \
                        dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/vae-epoch=189-train_loss=0.00.ckpt\'

# python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.beta1=1e-6 \
#                         dataset.ddpm.model.beta2=0.06 \
#                         dataset.ddpm.model.n_timesteps=100 \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_1' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1,2,3\' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/ddpmv2-cifar10_form1_scale=[-1,1]_28thDec_wavegrad_nc-epoch=899-loss=0.0293.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=128 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_form1_cont_samples_fixedddpm_30thDec_mainfig/\' \
#                         dataset.ddpm.evaluation.n_samples=50000 \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.variance='fixedsmall' \
#                         dataset.ddpm.evaluation.save_vae=False \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.persistent_buffers=False \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\'
