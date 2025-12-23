#!/usr/bin/env bash

# Usage:
#   bash bash_scripts/infer_single.sh path/to/audio.wav "your caption"
#
# If no arguments are provided, use default example values.


AUDIO_PATH="example/Ym8O802VvJes.wav"
CAPTION="Mix in dog barking in the middle."


CUDA_VISIBLE_DEVICES=0 python infer.py \
    +audio_path="$AUDIO_PATH" \
    +caption="$CAPTION" \
    +ckpt_dir=ckpt/mmedit \
    +use_best=true \