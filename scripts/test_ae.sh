python main/test.py reconstruct --device gpu:0 \
                                --dataset celeba \
                                --image-size 64 \
                                --save-path ~/vae_celeba64_recons/ \
                                --write-mode numpy \
                                ~/vae_celeba64_alpha\=1.0/checkpoints/vae-celeba64_alpha\=1.0-epoch\=245-train_loss\=0.0000.ckpt \
                                ~/datasets/img_align_celeba/

# python main/test.py reconstruct --device gpu:0 \
#                            --num-samples 16 \
#                            --save-path ~/vae_alpha_1_0_samples/ \
#                            ~/checkpoints_old/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
#                            ~/datasets/CelebAMask-HQ/