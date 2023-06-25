#!/bin/bash
while getopts t:v:w:n: flag
do
    case "${flag}" in
        t) train_dir=${OPTARG};;
        v) val_dir=${OPTARG};;
        W) wandb_api_key=${OPTARG};;
        n) num_devices=${OPTARG};;
    esac
done
ulimit -n 4096
python main.py --train_dir $train_dir --val_dir $val_dir --wandb_api_key $wandb_api_key --device gpu --num_devices $num_devices