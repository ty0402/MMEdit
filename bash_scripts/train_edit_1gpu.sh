export http_proxy=http://taoye:iwGOqbNPeiu28u47afBOYh7HC8Ifx7vgmIZeIlffRRjRul2krjObXXXT0ZGm@10.1.20.50:23128/ ; export https_proxy=http://taoye:iwGOqbNPeiu28u47afBOYh7HC8Ifx7vgmIZeIlffRRjRul2krjObXXXT0ZGm@10.1.20.50:23128/
# export PATH=/mnt/petrelfs/taoye/.local/bin:$PATH
# sed -i 's|^#![[:space:]]*/usr/bin/python.*|#!/usr/local/lib/miniconda3/bin/python|' /mnt/petrelfs/taoye/.local/bin/accelerate

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file /mnt/petrelfs/taoye/workspace/MMEdit/configs/accelerate/nvidia/1gpu.yaml /mnt/petrelfs/taoye/workspace/MMEdit/train.py \
    epochs=100 \
    data@data_dict=edit \
    train_dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    model=mmdit \
    exp_dir=/mnt/petrelfs/taoye/workspace/editing/exp/test \
    exp_name=mmedit \
    optimizer.lr=3e-5 \
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/opencpop/checkpoints/epoch_51
    # ~trainer.wandb_config \
