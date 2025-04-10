#! /bin/bash
set -e 



DEFAULT_GPUS_PER_NODE=8
DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT=25001

#######################################################################
# slurm related args, you can ignore them if you are not using slurm
# Get the slurm job name
SLURM_JOB_NAME=${SLURM_JOB_NAME:-"default_job"}
# Split SLURM_JOB_NAME by "/" and get the last segment
if [[ "$SLURM_JOB_NAME" == *"/"* ]]; then
    # Extract the last segment after the last "/"
    SLURM_JOB_NAME=$(echo "$SLURM_JOB_NAME" | rev | cut -d'/' -f1 | rev)
fi
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"

NNODES=${SLURM_JOB_NUM_NODES:-1}
echo "NNODES = $NNODES"

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "NODES = $NODES"

NODE_RANK=${SLURM_PROCID:-0}
echo "NODE_RANK = $NODE_RANK"

GPUS_PER_NODE=${SLURM_JOB_GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}
echo "GPUS_PER_NODE = $GPUS_PER_NODE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}
echo "MASTER_ADDR = $MASTER_ADDR"

MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}
echo "MASTER_PORT = $MASTER_PORT"
#######################################################################

# Disable wandb logging
# export WANDB_MODE="disabled"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_$SLURM_JOB_NAME.txt"

# model_name_or_path=Qwen/Qwen2-VL-2B-Instruct
# model_name_or_path=/home/ligengz/workspace/VILA-main/NVILA-Lite-2B-hf-preview
model_name_or_path=Efficient-Large-Model/NVILA-Lite-2B-hf-preview

echo "model_name_or_path = $model_name_or_path"

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo.py \
    --output_dir ./logs/$SLURM_JOB_NAME \
    --model_name_or_path $model_name_or_path \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --save_total_limit 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name $SLURM_JOB_NAME \
    --save_steps 10 \
    --save_only_model true \
    --learning_rate 1e-8 \
    --num_generations 2   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

exit 0 

NNODES=1
JOB_NAME="VILA-Lite-small-lr-2B-GRPO-CLEVR-70k-nodes_${NNODES}"
srun --account $VILA_SLURM_ACCOUNT --partition $VILA_SLURM_PARTITION \
    --job-name $VILA_SLURM_ACCOUNT:train/$JOB_NAME \
    --nodes $NNODES \
    --gpus-per-node 8 \
    --time 4:00:00 \
    --exclusive \
    --output ./logs/$JOB_NAME/slurm/%J.out \
    --error ./logs/$JOB_NAME/slurm/%J.err \
    bash run_vila.sh
