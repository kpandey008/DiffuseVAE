# python test_ddpm.py interpolate-vae --n-steps 1000 \
#                                 --device gpu:3 \
#                                 --save-path ~/cond-inference/ \
#                                 --seed 99 \
#                                 --n-interpolate 10 \
#                                 ~/checkpoints/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/checkpoints/celebahq128/ddpm_celebahq128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt


python test_ddpm.py interpolate-ddpm --n-steps 1000 \
                                --device gpu:3 \
                                --save-path ~/cond-inference/ \
                                --seed 99 \
                                --n-interpolate 10 \
                                ~/checkpoints/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/checkpoints/celebahq128/ddpm_celebahq128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt