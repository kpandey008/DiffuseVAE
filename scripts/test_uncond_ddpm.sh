ulimit -n 2048
python test_ddpm.py sample --n-steps 300 \
                            --device gpu:3 \
                            --save-path ~/ddpm_samples_cifar10_nsamples10k_uncond/ \
                            --sample-prefix '3' \
                            --num-samples 2500 \
                            --seed 3 \
                            --image-size 32 \
                            --batch-size 32 \
                            --n-workers 2 \
                            --checkpoints "" \
                            ~/checkpoints/cifar10/ddpm_cifar10_uncond/ddpmv2-epoch\=1006-loss\=0.0662.ckpt