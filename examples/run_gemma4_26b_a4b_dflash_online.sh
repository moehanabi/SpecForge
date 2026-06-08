#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train dflash for gemma4-26b-a4b
NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path google/gemma-4-26b-a4b-it \
    --draft-config-path $ROOT_DIR/configs/gemma4-26b-a4b-dflash.json \
    --train-data-path $ROOT_DIR/outputs/dataset/ultrachat_regen_gemma4_preformatted.jsonl \
    --is-preformatted \
    --output-dir $ROOT_DIR/outputs/gemma4-26b-a4b-dflash \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template gemma-4 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --report-to tensorboard \
    --target-model-backend sglang \
    --block-size 16 \
    --mask-token-id 4 \
    --embedding-key model.language_model.embed_tokens.weight \
    --lm-head-key model.language_model.lm_head.weight
