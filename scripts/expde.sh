# python main/extract_latents.py extract --device gpu:0 \
#                                 --dataset-name celeba \
#                                 --image-size 64 \
#                                 --save-path ~/celeba64_latents/ \
#                                 '/data1/kushagrap20/checkpoints/celeba64/vae_celeba64_alpha=1.0/checkpoints/vae-celeba64_alpha=1.0-epoch=245-train_loss=0.0000.ckpt' \
#                                 ~/datasets/img_align_celeba/

# Fit GMM CMHQ-128
# python main/expde.py fit-gmm ~/cmhq128_latents/latents_celebamaskhq.npy --save-path '/data1/kushagrap20/cmhq128_latents/gmm_z/' --n-components 150

# Fit GMM CIFAR-10
# python main/expde.py fit-gmm ~/cifar10_latents/latents_cifar10.npy --save-path '/data1/kushagrap20/cifar10_latents/gmm_z/' --n-components 100

# Fit GMM CelebA-64
python main/expde.py fit-gmm ~/celeba64_latents/latents_celeba.npy --save-path '/data1/kushagrap20/celeba64_latents/gmm_z/' --n-components 50
