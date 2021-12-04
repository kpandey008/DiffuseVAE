# python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=False \
#                         dataset.ddpm.evaluation.seed=1234 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/content/drive/MyDrive/ddpm_celebamaskhq_26thOct_form2_scale[01]/ddpmv2-celebamaskhq_26thOct_form2_scale01-epoch=250-loss=0.0022.ckpt\' \
#                         dataset.ddpm.evaluation.type='form2' \
#                         dataset.ddpm.evaluation.temp=0.1 \
#                         dataset.ddpm.evaluation.batch_size=4 \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_path=\'/content/ddpm_test_form2_temp=0.1\' \
#                         dataset.ddpm.evaluation.n_samples=32 \
#                         dataset.ddpm.evaluation.n_steps=200 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/content/drive/MyDrive/project_vaedm/celebahq-mask/vae_chkpt_celebahq/vae-epoch=189-train_loss=0.00.ckpt\'


# python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.n_heads=4 \
#                         dataset.ddpm.model.n_residual=3 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/ddpmv2-cifar10_form1_scale=[-1,1]_18thNov_sota_nres3_nheads4-epoch=1199-loss=0.0228.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=128 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_form1_updated_architecture_2_normtest\' \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=False \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\'


python main/eval/ddpm/sample_cond.py +dataset=celeba64/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.evaluation.seed=3 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_3' \
                        dataset.ddpm.evaluation.device=\'gpu:3\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/ddpm_celeba64_21stNov_form1_sota/checkpoints/ddpmv2-celebamaskhq_celeba64_21stNov_form1_sota-epoch=656-loss=0.0090.ckpt\' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_celeba64_form1_sota_50k\' \
                        dataset.ddpm.evaluation.n_samples=12500 \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=False \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/vae_celeba64_alpha=1.0/checkpoints/vae-celeba64_alpha=1.0-epoch=245-train_loss=0.0000.ckpt\'