
# for text spotting
python main.py \
       --eval \
       --data_root ./data/text_spotting_datasets/ \
       --val_dataset totaltext_val \
       --output_folder ./output/test/totaltext/ \
       --tfm_pre_norm \
       --test_max_size 1920 \
       --test_min_size 1920 \
       --use_fpn \
       --global_prob 1 \
       --visualize \
       --use_char_window_prompt \
       # --resume ckpt_path

# for kie
python main.py \
       --eval \
       --data_root ./data/text_spotting_datasets/ \
       --val_dataset cord_val \
       --output_folder ./output/test/cord \
       --tfm_pre_norm \
       --test_max_size 1920 \
       --test_min_size 1920 \
       --use_fpn \
       --vie_categories 29 \
       --global_prob 1 \
       --infer_vie \
       --use_char_window_prompt \
       # --resume ckpt_path       
