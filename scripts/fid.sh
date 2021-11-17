ulimit -n 2048
# python -m pytorch_fid --device cuda:0 ~/datasets/downsampled/CelebA-HQ-128/ ~/ddpm_samples_celebahq_128_nsamples10k_form1/500/images/
# /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_pytorch_train_32.npz
python third_party/pytorch_fid/fid.py --device cuda:3 --mode1 np --mode2 np /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_pytorch_train_32.npz /data/kushagrap20/ddpm_samples_cifar10_nsamples10k_uncond_fixedlarge_test2/1000/images/

# python eval/fid.py compute-fid-from-samples --num-batches 500 \
#                                             /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_tensorflow_train_32.npz \
#                                             /data/kushagrap20/ddpm_samples_cifar10_nsamples50k_uncond_fixedlarge_test2/1000/images