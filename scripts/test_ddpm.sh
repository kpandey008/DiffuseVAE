python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
                        dataset.ddpm.data.norm=False \
                        dataset.ddpm.evaluation.seed=1234 \
                        dataset.ddpm.evaluation.chkpt_path=\'/content/drive/MyDrive/ddpm_celebamaskhq_26thOct_form2_scale[01]/ddpmv2-celebamaskhq_26thOct_form2_scale01-epoch=250-loss=0.0022.ckpt\' \
                        dataset.ddpm.evaluation.type='form2' \
                        dataset.ddpm.evaluation.temp=0.1 \
                        dataset.ddpm.evaluation.batch_size=4 \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_path=\'/content/ddpm_test_form2_temp=0.1\' \
                        dataset.ddpm.evaluation.n_samples=32 \
                        dataset.ddpm.evaluation.n_steps=200 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/content/drive/MyDrive/project_vaedm/celebahq-mask/vae_chkpt_celebahq/vae-epoch=189-train_loss=0.00.ckpt\'