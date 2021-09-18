ulimit -n 2048
python test_ddpm.py sample-cond --n-steps 500 \
                                --device gpu:0,1,2 \
                                --save-path ~/cond_inference_form1/ \
                                --num-samples 4 \
                                --compare False \
                                --seed 0 \
                                --batch-size 1 \
                                --n-workers 1 \
                                --checkpoints "" \
                                ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/ddpm_128_form1/ddpmv2-epoch\=801-loss\=0.0434.ckpt

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