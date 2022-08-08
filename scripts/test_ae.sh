# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset celeba \
#                                 --image-size 64 \
#                                 --save-path ~/vae_celeba64_recons/ \
#                                 --write-mode numpy \
#                                 ~/vae_celeba64_alpha\=1.0/checkpoints/vae-celeba64_alpha\=1.0-epoch\=245-train_loss\=0.0000.ckpt \
#                                 ~/datasets/img_align_celeba/

# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset ffhq \
#                                 --image-size 128 \
#                                 --num-samples 64 \
#                                 --save-path ~/vae_samples_ffhq128_deletem_recons/ \
#                                 --write-mode image \
#                                 /data1/kushagrap20/vae_ffhq128_11thJune_alpha\=1.0/checkpoints/vae-ffhq128_11thJune_alpha\=1.0-epoch\=496-train_loss\=0.0000.ckpt \
#                                 ~/datasets/ffhq/

# python main/test.py sample --device gpu:0 \
#                             --image-size 32 \
#                             --seed 0 \
#                             --num-samples 50000 \
#                             --save-path ~/vae_samples_cifar10_deleteme/ \
#                             --write-mode image \
#                             512 \
#                             /data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt \

# python main/test.py sample --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 64 \
#                             --save-path ~/vae_samples_ffhq128_deletem/ \
#                             --write-mode image \
#                             1024 \
#                             /data1/kushagrap20/vae_ffhq128_11thJune_alpha\=1.0/checkpoints/vae-ffhq128_11thJune_alpha\=1.0-epoch\=496-train_loss\=0.0000.ckpt \


# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset afhq \
#                                 --image-size 128 \
#                                 --save-path ~/reconstructions/afhq_reconsv2/ \
#                                 --write-mode numpy \
#                                 ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \
#                                 ~/datasets/afhq/

# python main/test.py sample --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 64 \
#                             --save-path ~/afhq_vae_samples1/ \
#                             --write-mode image \
#                             1024 \
#                             ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \

# python main/test.py reconstruct --device gpu:0 \
#                            --dataset celebamaskhq \
#                            --num-samples 16 \
#                            --save-path ~/vae_alpha_1_0_samples/ \
#                            ~/checkpoints/cmhq/vae-epoch\=189-train_loss\=0.00.ckpt \
#                            ~/datasets/CelebAMask-HQ/