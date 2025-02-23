#!/bin/bash

# ==== Set environment variables ==== #
WORK_DIR="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/src/open-r1-multimodal"
cd $WORK_DIR

set -x

# ==== Set training parameters ==== #
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-2}

# ==== Set output directory ==== #
RUN_NAME="qwen_2_vl_2b_vllm_grpo_r1m_8k_gy_lr1e-6_bs1_ga2_ep2_torchrun"
OUTPUT_DIR="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/output/${RUN_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# ==== Select model, dataset, and deepspeed configuration ==== #
MODEL_NAME="/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-2B-Instruct"
DATASET_NAME="/apdcephfs_gy2/share_302735770/stephenruan/data/leonardPKU___clevr_cogen_a_train"
# DATASET_NAME="/apdcephfs_gy2/share_302735770/stephenruan/data/lmms-lab___multimodal-open-r1-8k-verified"

# ==== Node configuration ==== #
# Number of training nodes
NUM_TRAIN_NODES=1
# Number of vLLM sampler nodes
NUM_VLLM_NODES=1
# Total number of nodes
IFS=',' read -ra NODE_ARRAY <<< "$NODE_IP_LIST"
TOTAL_NODES=${#NODE_ARRAY[@]}

# ==== Master node configuration ==== #
MASTER_ADDR=$(echo ${NODE_ARRAY[0]} | cut -d ':' -f 1)  # Use the first node as the master node
MASTER_PORT=12345

# ==== Launch all nodes using pssh ==== #
# Generate the launch command for each node
LAUNCH_COMMANDS=()
for ((i=0; i<TOTAL_NODES; i++)); do
    if [ $i -lt $NUM_TRAIN_NODES ]; then
        # Training node command
        COMMAND="torchrun --nproc_per_node 8 \
            --nnodes $TOTAL_NODES \
            --node_rank $i \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            src/open_r1/grpo_vllm.py \
            --output_dir $OUTPUT_DIR \
            --model_name_or_path $MODEL_NAME \
            --dataset_name $DATASET_NAME \
            --max_prompt_length 8192 \
            --max_completion_length 8192 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --num_generations 7 \
            --logging_steps 1 \
            --bf16 \
            --report_to tensorboard \
            --gradient_checkpointing true \
            --attn_implementation flash_attention_2 \
            --max_pixels 401408 \
            --save_total_limit 3 \
            --num_train_epochs 2 \
            --run_name $RUN_NAME"
    else
        # vLLM sampler node command
        COMMAND="torchrun --nproc_per_node 1 \
            --nnodes $TOTAL_NODES \
            --node_rank $i \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            src/open_r1/grpo_vllm.py \
            --output_dir $OUTPUT_DIR \
            --model_name_or_path $MODEL_NAME \
            --dataset_name $DATASET_NAME \
            --max_prompt_length 8192 \
            --max_completion_length 8192 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --num_generations 7 \
            --logging_steps 1 \
            --bf16 \
            --report_to tensorboard \
            --gradient_checkpointing true \
            --attn_implementation flash_attention_2 \
            --max_pixels 401408 \
            --save_total_limit 3 \
            --num_train_epochs 2 \
            --run_name $RUN_NAME \
            --is_vllm_sampler true"
    fi
    LAUNCH_COMMANDS+=("$COMMAND")
done

# Use pssh to launch commands on all nodes
for ((i=0; i<TOTAL_NODES; i++)); do
    NODE_IP=$(echo ${NODE_ARRAY[$i]} | cut -d ':' -f 1)  # Get the IP of the node
    echo "Launching on node $NODE_IP: ${LAUNCH_COMMANDS[$i]}"
    pssh -H $NODE_IP -i "${LAUNCH_COMMANDS[$i]}" &
done

# Wait for all background processes to complete
wait
echo "All nodes have been launched."