
work_dir=/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/src/open-r1-multimodal
cd $work_dir

set -x

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-2}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export PYTHONPATH=src:$PYTHONPATH

# ==== 设置输出路径 ==== #
RUN_NAME="qwen_2_vl_2b_grpo_clevr_70k_gy_lr1e-6_bs1_ga2_ep2_torchrun"
OUTPUT_DIR="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/output/${RUN_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
# ==== 选择模型、数据集和deepspeed配置 ==== #
MODEL_NAME="/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-2B-Instruct"
# MODEL_NAME="/apdcephfs_sh8/share_301266059/stephenruan/code/src/Qwen2-VL-7B-Instruct"
DATASET_NAME="/apdcephfs_gy2/share_302735770/stephenruan/data/leonardPKU___clevr_cogen_a_train"
DS_CONFIG="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/src/open-r1-multimodal/local_scripts/zero1.json"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${OUTPUT_DIR}/debug_log_2b.txt"


torchrun \
    --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to tensorboard \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"