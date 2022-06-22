python train.py \
dset=noisy_clean \
experiment_name=h48u4s4_conditioning_lrn \
hidden=48 \
stride=4 \
resample=4 \
features_dim=768 \
features_dim_for_conditioning=768 \
include_ft=True \
get_ft_after_lstm=False \
use_as_conditioning=True \
use_as_supervision=False \
layers=[0,1,2,3,4,5,6,7,8,9,10,11,12] \
learnable=True \
do_enhance=False \
ddp=True \
batch_size=16