#!/bin/bash

# AGGREGATOR_PATH=your/path/to/Prism
# RAW_FEAT_PATH=your/path/to/raw/patch/features
# DATA_PATH=your/path/for/training

AGGREGATOR_PATH=/home2/hyun/REG/repo/ToE_HistoREG/Prism
RAW_FEAT_PATH=/home2/hyun/REG/data/20x_224px_0px_overlap/features_virchow_pt
# DATA_PATH=/home2/hyun/REG/repo/ToE_HistoREG/data
DATA_PATH=/home2/hyun/REG/proposed/exp_bhl_v3/data


ALPHA="0.1"
LAM="0.01" 
PROBE_TYPE="mlp"

CUDA_VISIBLE_DEVICES="0" python ./train_prism_hier_cls.py \
    --aggregator_path ${AGGREGATOR_PATH} \
    --raw_feature_path ${RAW_FEAT_PATH} \
    --val_anno_path ${DATA_PATH}/val_samples.json \
    --feature_fn ${DATA_PATH}/feature_cache/features_all_trident.pt \
    --records_path ${DATA_PATH}/tree_samples \
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
    --use_wandb True
