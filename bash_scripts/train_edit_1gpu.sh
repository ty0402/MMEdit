
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml train.py \
    epochs=100 \
    data@data_dict=example \
    train_dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    model=mmdit \
    exp_dir=exp/test \
    exp_name=mmedit \
    optimizer.lr=3e-5 \
    