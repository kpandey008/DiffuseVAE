ulimit -n 2048
python test_ddpm.py sample --n-steps 1000 \
                            --device gpu:3 \
                            --sample-prefix '3' \
                            --seed 3 \
                            --image-size 32 \
                            --save-path ~/ddpm_samples_cifar10_nsamples10k_uncond_fixedlarge_np/ \
                            --num-samples 2500 \
                            --batch-size 64 \
                            --n-workers 2 \
                            --checkpoints "" \
                            --save-mode numpy \
                            ~/checkpoints/cifar10/ddpm_cifar10_uncond/ddpmv2-epoch\=2031-loss\=0.0761.ckpt

# python test_ddpm.py sample --n-steps 500 \
#                             --device gpu:3 \
#                             --sample-prefix '3' \
#                             --seed 3 \
#                             --save-path ~/ddpm_samples_celebahq128_nsamples5k_uncond/ \
#                             --num-samples 1250 \
#                             --image-size 128 \
#                             --batch-size 16 \
#                             --n-workers 3 \
#                             --checkpoints "" \
#                             ~/checkpoints/celebahq128/ddpm_celebahq128_uncond/ddpmv2-epoch\=795-loss\=0.0494.ckpt
