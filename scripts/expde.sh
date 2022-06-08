# python main/extract_latents.py extract --device gpu:0 \
#                                 --dataset-name cifar10 \
#                                 --image-size 32 \
#                                 --save-path ~/cifar10_latents/ \
#                                 '/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt' \
#                                 ~/datasets/

python main/expde.py fit-gmm ~/cifar10_latents/latents_cifar10.npy --save-path '/data1/kushagrap20/cifar10_latents/gmm_z/' --n-components 50