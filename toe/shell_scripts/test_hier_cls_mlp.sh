#!/bin/bash

PROBE_TYPE="mlp"

CUDA_VISIBLE_DEVICES="0" python ./test_prism_hier_cls.py \
    --aggregator_path ./Prism \
    --raw_feature_path {PATCH_PATH} \
    --feature_fn ./data/feature_cache/features_test_phase2.pt \
    --test_output_path ./results/exp_${PROBE_TYPE}_test_phase2 \
    --ckpt_path ./ckpt/ckpt_${PROBE_TYPE}.pt \
    --probe_type mlp \
    --threshold 0.5