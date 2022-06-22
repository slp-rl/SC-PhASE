python train.py \
dset=noisy_clean \
experiment_name=h48u4s4_supervision \
hidden=48 \
stride=4 \
resample=4 \
features_dim=768 \
features_dim_for_conditioning=768 \
include_ft=False \
get_ft_after_lstm=False \
use_as_conditioning=False \
use_as_supervision=True \
layers=[6] \
learnable=False \
do_enhance=False \
ddp=True \
batch_size=16