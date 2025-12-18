AUDIO_PATH="example/Ym8O802VvJes.wav"
CAPTION="Mix in dog barking in the middle."


CUDA_VISIBLE_DEVICES=1 python infer.py \
    +audio_path="$AUDIO_PATH" \
    +caption="$CAPTION" \
    +ckpt_dir=ckpt/mmedit \
    +use_best=true \