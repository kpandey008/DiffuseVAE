ulimit -n 2048

# CIFAR-10
# python test_ddpm.py sample-cond --n-steps 700 \
#                                 --device gpu:3 \
#                                 --sample-prefix '3' \
#                                 --seed 3 \
#                                 --z-dim 512 \
#                                 --image-size 32 \
#                                 --save-path ~/ddpm_samples_cifar10_nsamples5k_form1/ \
#                                 --num-samples 2500 \
#                                 --compare False \
#                                 --batch-size 64 \
#                                 --n-workers 2 \
#                                 --checkpoints "" \
#                                 ~/checkpoints/cifar10/cifar10_ae/checkpoints/vae-epoch\=500-train_loss\=0.00.ckpt \
#                                 ~/checkpoints/cifar10/ddpm_cifar10_form1/ddpmv2-epoch=970-loss=0.0889.ckpt

# python test_ddpm.py sample-cond --n-steps 500 \
#                                 --device gpu:0,1,2,3 \
#                                 --save-path ~/ddpm_samples_celebahq_128_nsamples5k_form1/ \
#                                 --num-samples 5000 \
#                                 --compare False \
#                                 --seed 0 \
#                                 --batch-size 16 \
#                                 --n-workers 8 \
#                                 --checkpoints "" \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt

# python test_ddpm.py sample-cond --n-steps 500 \
#                                 --device gpu:0,1,2,3 \
#                                 --save-path ~/ddpm_samples_celebahq_128_nsamples5k_form1_folder2/ \
#                                 --num-samples 5000 \
#                                 --compare False \
#                                 --seed 1 \
#                                 --batch-size 16 \
#                                 --n-workers 8 \
#                                 --checkpoints "" \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt

# python test_ddpm.py generate-recons --n-steps 700 \
#                                 --device gpu:0,1,2,3 \
#                                 --save-path ~/cond_inference_form1/ \
#                                 --seed 0 \
#                                 --num-samples 8 \
#                                 --batch-size 2 \
#                                 --compare True \
#                                 --image-size 128 \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt \
#                                 ~/vaedm/reconstructions/