
# Pretrain Stage 1
python -m torch.distributed.launch \
        --nproc_per_node=1 \
       main.py \
       --data_root ./data/text_spotting_datasets/ \
       --output_folder ./output/pretrain/stage1/ \
       --train_dataset totaltext_train mlt_train ic13_train ic15_train syntext1_train syntext2_train \
       --lr 0.0005 \
       --max_steps 400000 \
       --warmup_steps 5000 \
       --checkpoint_freq 10000 \
       --batch_size 6 \
       --tfm_pre_norm \
       --train_max_size 768 \
       --rec_loss_weight 2 \
       --use_fpn \
       --use_char_window_prompt

# Pretrain Stage 2
python -m torch.distributed.launch \
        --nproc_per_node=1 \
       main.py \
       --data_root ./data/text_spotting_datasets/ \
       --output_folder ./output/pretrain/stage2/ \
       --train_dataset totaltext_train mlt_train ic13_train ic15_train syntext1_train syntext2_train \
       --lr 0.00025 \
       --max_steps 200000 \
       --warmup_steps 5000 \
       --checkpoint_freq 10000 \
       --batch_size 2 \
       --tfm_pre_norm \
       --train_max_size 1920 \
       --train_min_size 1600 1504 1408 1312 1216 1120 \
       --rec_loss_weight 2 \
       --use_fpn \
       --use_char_window_prompt \
#        --resume ckpt_path of pretrain stage 1

# Text spotting finetune
python -m torch.distributed.launch \
        --nproc_per_node=1 \
       main.py \
       --data_root ./data/text_spotting_datasets/ \
       --output_folder ./output/finetune/totaltext/ \
       --train_dataset totaltext_train \
       --lr 0.00025 \
       --max_steps 10000 \
       --warmup_steps 5000 \
       --checkpoint_freq 1000 \
       --batch_size 2 \
       --tfm_pre_norm \
       --train_max_size 1920 \
       --train_min_size 1600 1504 1408 1312 1216 1120 \
       --rec_loss_weight 2 \
       --use_fpn \
       --use_char_window_prompt \
#        --resume ckpt_path of pretrain stage 2

# VIE finetune
python -m torch.distributed.launch \
        --nproc_per_node=1 \
       main.py \
       --data_root ./data/text_spotting_datasets/ \
       --output_folder ./output/finetune/cord/ \
       --train_dataset cord_train \
       --lr 0.00025 \
       --max_steps 100000 \
       --warmup_steps 5000 \
       --checkpoint_freq 1000 \
       --batch_size 2 \
       --tfm_pre_norm \
       --train_max_size 1920 \
       --train_min_size 1600 1504 1408 1312 1216 1120 \
       --rec_loss_weight 2 \
       --use_fpn \
       --use_char_window_prompt \
       --vie_categories 29 \
       --global_prob 1 \
       --train_vie \
#        --resume ckpt_path of pretrain stage 2
