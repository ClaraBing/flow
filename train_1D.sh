#!/bin/bash

project='flow'
dataset='Gaussian1D'
layer_type='fc'
fdata='./data/synthetic/d10_std_train.npy'

lr=0.1
wd=1e-5
bt=128
use_val=1

wb_name="glow_G1D_lr$lr_wd$wd_bt$bt"

WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset=$dataset \
  --layer-type=$layer_type \
  --lr=$lr \
  --wd=$wd \
  --batch-size=$bt \
  --use-val=$use_val \
  --project=$project \
  --wb-name=$wb_name

