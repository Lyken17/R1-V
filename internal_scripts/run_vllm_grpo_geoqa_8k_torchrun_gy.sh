work_dir=/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/src/r1-v
cd $work_dir

set -x

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-8}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export PYTHONPATH=src:$PYTHONPATH

# ==== 设置输出路径 ==== #
RUN_NAME="qwen_2_vl_7b_vllm_grpo_geoqa_8k_gy_lr1e-6_bs1_ga8_ep2_torchrun"
OUTPUT_DIR="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/output/${RUN_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
# ==== 选择模型、数据集和deepspeed配置 ==== #
# MODEL_NAME="/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-2B-Instruct"
MODEL_NAME="/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-7B-Instruct"
# MODEL_NAME="/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2.5-VL-3B-Instruct"
DATASET_NAME="/apdcephfs_gy2/share_302735770/stephenruan/data/leonardPKU___geoqa_r1_v_train_8_k"
DS_CONFIG="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/internal_scripts/zero1.json"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${OUTPUT_DIR}/debug_log.txt"   # Only valid in single machine mode


torchrun \
    --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --max_prompt_length 4096 \
    --max_completion_length 2048 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to tensorboard \
    --use_vllm true \
    --temperature 1.0 \
    --num_generations 8 \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.8 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"