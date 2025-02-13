hang_dir=/apdcephfs_sh8/share_301266059/stephenruan/code/hang
work_dir=/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V/src/open-r1-multimodal
# ==== 先确保hang进程已经kill ==== #
cd $hang_dir
bash get_pssh_hosts.sh
pssh -i -t 0 -h pssh.hosts "ps -ef | grep python3 | grep -v grep | awk '{print \$2}' | xargs kill -9"
cd $work_dir

set -x

# ==== 配置机器参数和batch_size ==== #
pssh_hosts_path=${hang_dir}/pssh.hosts
# 若手动设置则在这里更改相关参数
HOSTFILE_DIR=$work_dir/"hostfiles"
if [ ! -d "$HOSTFILE_DIR" ]; then
    mkdir -p "$HOSTFILE_DIR"
fi
DEFAULT_HOSTFILE="${HOSTFILE_DIR}/hostfile_default"
DEFAULT_NUM_NODES=1
DEFAULT_MASTER_ADDR="127.0.0.1"

dist_info_choice="auto"
# dist_info_choice="manual"
# 根据 dist_info_choice 选择自动或手动
if [ "$dist_info_choice" == "auto" ]; then
    # 获取 launcher_ip
    launcher_ip=$(head -n 1 $pssh_hosts_path)

    # 生成 hostfile_{launcher_ip}
    hostfile_name="${HOSTFILE_DIR}/hostfile_${launcher_ip}"
    awk '{print $1 " slots=8"}' $pssh_hosts_path > "$hostfile_name"

    # 获取 num_nodes
    num_nodes=$(wc -l < $hostfile_name)

    # 获取 master_addr
    master_addr=$launcher_ip
else
    # 使用默认值
    hostfile_name=$DEFAULT_HOSTFILE
    num_nodes=$DEFAULT_NUM_NODES
    master_addr=$DEFAULT_MASTER_ADDR
fi

HOSTFILE=$hostfile_name
NUM_NODES=$num_nodes
MASTER_ADDR=$master_addr

GPUS=${GPUS:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-2}

# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH=src:$PYTHONPATH
export MASTER_PORT=15345
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# ==== 选择NCCL参数组 ==== #
case_flag="rdma"
# case_flag="normal"
if [ "$case_flag" == "rdma" ]; then
    echo "# ===== Using nccl parameters for machines supported RDMA ===== #"
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
    export NCCL_DEBUG=WARN
    export NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1
elif [ "$case_flag" == "normal" ]; then
    echo "# ===== Using nccl parameters for machines not supported RDMA ===== #"
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=bond1
    export NCCL_P2P_DISABLE=1
    export NCCL_DEBUG=INFO
else
    echo "Invalid case flag. Please set case_flag to 'normal' or 'rdma'."
    exit 1
fi

# ==== 设置输出路径 ==== #
RUN_NAME="qwen2_vl_2b_vllm_grpo_clevr_70k_gy_lr1e-6_bs1_ga2_ep2"
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

deepspeed \
    --hostfile ${HOSTFILE} \
    --num_nodes ${NUM_NODES} \
    --master_addr ${MASTER_ADDR} \
    --num_gpus ${GPUS} \
    --master_port ${MASTER_PORT} \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --max_prompt_length 1024 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to tensorboard \
    --deepspeed ${DS_CONFIG} \
    --use_vllm true \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"

# ==== 训练中断或结束后挂起 ==== #
cd /apdcephfs_sh8/share_301266059/stephenruan/code/hang
bash occupy.sh