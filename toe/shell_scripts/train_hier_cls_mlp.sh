#!/bin/bash

ALPHA="0.1"
LAM="0.01" 
PROBE_TYPE="mlp"

CUDA_VISIBLE_DEVICES="0" python ./train_prism_hier_cls.py \
    --aggregator_path ./Prism \
    --raw_feature_path {TRAIN_PATCH_PATH} \
    --val_anno_path ./data/val_samples.json \
    --feature_fn ./data/feature_cache/features_all.pt \
    --records_path ./data/tree_samples \
    --test_output_path ./results/exp_${PROBE_TYPE}_train \
    --ckpt_path ./ckpt/ckpt_${PROBE_TYPE}.pt \
    --project_name TOE_HistoREG \
    --exp_name all_organs_${PROBE_TYPE} \
    --probe_type ${PROBE_TYPE} \
    --batch_size 1024 \
    --max_epochs 1000 \
    --threshold 0.5 \
    --show_rep_period 50 \
    --val_period 100 \
    --alpha ${ALPHA} \
    --lam ${LAM} \
    --use_wandb False
