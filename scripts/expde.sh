# python main/extract_latents.py extract --device gpu:0 \
#                                 --dataset-name celebahq \
#                                 --image-size 256 \
#                                 --save-path ~/celebahq_latents/ \
#                                 '/data1/kushagrap20/vae-celebahq256_alpha=1.0_Jan31-epoch=499-train_loss=0.0000.ckpt' \
#                                 ~/datasets/celeba_hq/

# Fit GMM CMHQ-128
# python main/expde.py fit-gmm ~/cmhq128_latents/latents_celebamaskhq.npy --save-path '/data1/kushagrap20/cmhq128_latents/gmm_z/' --n-components 150

# Fit GMM CIFAR-10
# python main/expde.py fit-gmm ~/cifar10_latents/latents_cifar10.npy --save-path '/data1/kushagrap20/cifar10_latents/gmm_z/' --n-components 100

# Fit GMM CelebA-64
# python main/expde.py fit-gmm ~/celeba64_latents/latents_celeba.npy --save-path '/data1/kushagrap20/celeba64_latents/gmm_z/' --n-components 50

# Fit GMM AFHQ-256 Dogs
# python main/expde.py fit-gmm ~/afhq256_dog_latents/latents_afhq.npy --save-path '/data1/kushagrap20/afhq256_dog_latents/gmm_z/' --n-components 100

# Fit GMM CelebA-HQ-256
# python main/expde.py fit-gmm ~/celebahq_latents/latents_celebahq.npy --save-path '/data1/kushagrap20/celebahq_latents/gmm_z/' --n-components 150