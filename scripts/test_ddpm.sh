python test_ddpm.py sample-cond --n-steps 700 \
                                --device gpu:3 \
                                --save-path ~/cond_inference_form1/ \
                                --num-samples 1 \
                                --compare True \
                                --seed 0 \
                                ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/ddpm_128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt

# python test_ddpm.py generate-recons --n-steps 700 \
#                                 --device gpu:3 \
#                                 --save-path ~/cond_inference_form1/ \
#                                 --seed 0 \
#                                 --n-samples 8 \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt \
#                                 ~/vaedm/reconstructions/