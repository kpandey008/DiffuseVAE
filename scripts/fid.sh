ulimit -n 2048
# python -m pytorch_fid --device cuda:0 ~/datasets/downsampled/CelebA-HQ-128/ ~/ddpm_samples_celebahq_128_nsamples10k_form1/500/images/
# /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_pytorch_train_32.npz
python third_party/pytorch_fid/fid.py --device cuda:0 --mode1 np --mode2 np /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_pytorch_train_32.npz /data/kushagrap20/ddpm_samples_cifar10_nsamples10k_uncond_fixedlarge_np/1000/images/