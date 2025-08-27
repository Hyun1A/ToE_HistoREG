#!/bin/bash

PROBE_TYPE="mlp"

CUDA_VISIBLE_DEVICES="0" python ./val_prism_hier_cls.py \
    --aggregator_path ./Prism \
    --raw_feature_path {TRAIN_PATCH_PATH} \
    --val_anno_path ./data/val_samples.json \
    --feature_fn ./data/feature_cache/features_all.pt \
    --records_path ./data/tree_samples \
    --test_output_path ./results/exp_${PROBE_TYPE}_train \
    --ckpt_path ./ckpt/ckpt_${PROBE_TYPE}.pt \
    --probe_type mlp \
    --threshold 0.5