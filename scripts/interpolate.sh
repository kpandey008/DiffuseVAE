python main/eval/ddpm/interpolate_ddpm.py +dataset=celebamaskhq128/test \
                            dataset.ddpm.data.norm=True \
                            dataset.ddpm.model.beta1=1e-6 \
                            dataset.ddpm.model.beta2=0.7 \
                            dataset.ddpm.model.n_timesteps=10 \
                            dataset.ddpm.evaluation.save_mode='image' \
                            dataset.ddpm.data.ddpm_latent_path=\'/data1/kushagrap20/cmhq_ddpm_latents.pt\' \
                            dataset.ddpm.evaluation.seed=99 \
                            dataset.ddpm.evaluation.temp=1.0 \
                            dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_4thNov_form2_scale[-11]_wavegrad_nc-epoch=251-loss=0.0158.ckpt\' \
                            dataset.ddpm.evaluation.type='form2' \
                            dataset.ddpm.evaluation.batch_size=1 \
                            dataset.ddpm.evaluation.device=\'gpu:0\' \
                            dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/interpolate_ddpm\' \
                            dataset.ddpm.evaluation.n_samples=1 \
                            dataset.ddpm.evaluation.n_steps=10 \
                            dataset.ddpm.evaluation.save_vae=True \
                            dataset.ddpm.evaluation.workers=1 \
                            dataset.ddpm.evaluation.persistent_buffers=False \
                            dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/vae-epoch=189-train_loss=0.00.ckpt\'