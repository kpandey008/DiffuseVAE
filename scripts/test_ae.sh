python main/test.py reconstruct --device gpu:0 \
                           --num-samples 16 \
                           --save-path ~/vae_alpha_0_1_samples/ \
                           ~/vae_alpha_0_1/checkpoints/vae-celebamaskhq_alpha_0_1-epoch\=999-train_loss\=0.0000.ckpt \
                           ~/datasets/CelebAMask-HQ/

# python main/test.py reconstruct --device gpu:0 \
#                            --num-samples 16 \
#                            --save-path ~/vae_alpha_1_0_samples/ \
#                            ~/checkpoints_old/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
#                            ~/datasets/CelebAMask-HQ/