# Set the path to save checkpoints
OUTPUT_DIR='results/vits'

# path to SSV2 annotation file (train.csv/val.csv/test.csv)
DATA_PATH='/scratch/datasets/ssv2/labels/label_csv/'
# path to pretrain model
MODEL_PATH='checkpoints/videomae_ssv2_vits.pth'

# batch_size can be adjusted according to number of GPUs
# VideoMAE: this script is for 32 GPUs (4 nodes x 8 GPUs)
# We: this script is for 2 GPUs
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --finetune ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 12 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    # --enable_deepspeed 
# 
