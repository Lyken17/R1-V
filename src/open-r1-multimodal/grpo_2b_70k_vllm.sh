export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_rerun.txt"
QWEN_PATH="/home/lilei/Qwen2-VL-2B-Instruct"
HF_DATASET="/home/lilei/Clevr_CoGenT_TrainA_70K" 
OUTPUT_DIR="/home/lilei/R1-V/checkpoints/Qwen2-VL-2B-GRPO-R1-70K" 

CUDA_VISIBLE_DEVICES="0,1,2,3,4" torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 400000 \
    --max_steps 13125 \
    --run_name Qwen2-VL-2B-GRPO-R1-70K-5GPUs-nG4-maxNew1K-3epoch \
    --save_steps 1000 \
    --save_only_model true
