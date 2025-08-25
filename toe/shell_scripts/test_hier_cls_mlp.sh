#!/bin/bash

# AGGREGATOR_PATH=your/path/to/Prism
# RAW_FEAT_PATH=your/path/to/raw/test/patch/test_features
# DATA_PATH=your/path/for/training

AGGREGATOR_PATH=/home2/hyun/REG/repo/ToE_HistoREG/Prism
RAW_FEAT_PATH=/home2/hyun/REG/data/test_data_phase2/features_virchow_pt
# DATA_PATH=/home2/hyun/REG/repo/ToE_HistoREG/data
DATA_PATH=/home2/hyun/REG/proposed/exp_bhl_v3/data


PROBE_TYPE="mlp"

CUDA_VISIBLE_DEVICES="0" python ./test_prism_hier_cls.py \
    --aggregator_path ${AGGREGATOR_PATH} \
    --raw_feature_path ${RAW_FEAT_PATH} \
    --feature_fn ${DATA_PATH}/feature_cache/features_test_phase2.pt \
    --test_output_path ./results/exp_${PROBE_TYPE}_test_phase2 \
    --ckpt_path ./ckpt/ckpt_${PROBE_TYPE}.pt \
    --probe_type mlp \
    --threshold 0.5