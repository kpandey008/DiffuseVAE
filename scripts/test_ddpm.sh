# python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_3' \
#                         dataset.ddpm.evaluation.device=\'gpu:3\' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_24thOct-epoch=259-loss=0.0054.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=2 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_form2_cmhq_temp=1.0_5k/\' \
#                         dataset.ddpm.evaluation.n_samples=8 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/vae-epoch=189-train_loss=0.00.ckpt\'

# python main/eval/ddpm/sample.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_3' \
#                         dataset.ddpm.evaluation.device=\'gpu:3\' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_1stNov_uncond_scale[-11]-epoch=268-loss=0.0021.ckpt\' \
#                         dataset.ddpm.evaluation.type='uncond' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=2 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_uncond_ddim_cmhq_temp=1.0_5k/\' \
#                         dataset.ddpm.evaluation.n_samples=8 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/vae-epoch=189-train_loss=0.00.ckpt\'


# python main/eval/ddpm/sample_cond.py +dataset=afhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.evaluation.seed=42 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1,2,3\' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/ddpm_afhq_13thDec_form1_scale[-11]/checkpoints/ddpmv2-afhq_13thDec_form1_scale[-11]-epoch=402-loss=0.0045.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=4 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/afhq_form1_samples/\' \
#                         dataset.ddpm.evaluation.n_samples=128 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/vae_afhq_alpha=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt\'


# python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,8\' \
#                         dataset.ddpm.model.n_residual=3 \
#                         dataset.ddpm.model.dim_mults=\'1,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/ddpm_cifar10_form1_scale=[-1,1]_7thApr_sota_nres=4_nheads=8_mults=1234/checkpoints/ddpmv2-cifar10_form1_scale=[-1,1]_18thNov_sota_nres3_nheads4-epoch=825-loss=0.0433.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_form1_newsota\' \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=False \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\'

python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.n_residual=4 \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.seed=3 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_3' \
                        dataset.ddpm.evaluation.device=\'gpu:3\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/ddpm_cifar10_form1_scale=[-1,1]_7thMay_sota_nres=4_nheads=8_mults=1222_dim=128/checkpoints/ddpmv2-cifar10_form1_scale=[-1,1]_7thMay_sota_nres=4_nheads=8_mults=1222_dim=128-epoch=742-loss=0.0150.ckpt\' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_form1_newsota_beta=0.2\' \
                        dataset.ddpm.evaluation.n_samples=2500 \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=False \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/compactvae-cifar10_30thApr_beta=0.2-epoch=499-train_loss=0.0000.ckpt\'


# python main/eval/ddpm/sample_cond.py +dataset=celeba64/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.evaluation.seed=3 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_3' \
#                         dataset.ddpm.evaluation.device=\'gpu:3\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/ddpm_celeba64_21stNov_form1_sota/checkpoints/ddpmv2-celebamaskhq_celeba64_21stNov_form1_sota-epoch=656-loss=0.0090.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_celeba64_form1_sota_50k\' \
#                         dataset.ddpm.evaluation.n_samples=12500 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=False \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/vae_celeba64_alpha=1.0/checkpoints/vae-celeba64_alpha=1.0-epoch=245-train_loss=0.0000.ckpt\'