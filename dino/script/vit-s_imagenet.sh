python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch vit_small \
--data_path /path/to/imagenet/train \
--output_dir ./log/dino/vit-s_imagenet \
