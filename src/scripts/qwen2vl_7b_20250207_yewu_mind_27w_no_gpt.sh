hang_dir=/apdcephfs_sh8/share_301266059/stephenruan/code/hang
# work_dir=/apdcephfs_sh8/share_301266059/boweijia/ocr_pro/InternVL/internvl_chat
work_dir=/apdcephfs_cq8/share_1611098/ruanzheng/code/src/Qwen2-VL-Finetune
# ==== 先确保hang进程已经kill ==== #
cd $hang_dir
bash get_pssh_hosts.sh
pssh -i -t 0 -h pssh.hosts "ps -ef | grep python3 | grep -v grep | awk '{print \$2}' | xargs kill -9"
cd $work_dir

set -x

# ==== 配置机器参数和batch_size ==== #
pssh_hosts_path=${hang_dir}/pssh.hosts
# 若手动设置则在这里更改相关参数
DEFAULT_HOSTFILE="hostfiles/hostfile_default"
DEFAULT_NUM_NODES=1
DEFAULT_MASTER_ADDR="127.0.0.1"

dist_info_choice="auto"
# dist_info_choice="manual"
# 根据 dist_info_choice 选择自动或手动
if [ "$dist_info_choice" == "auto" ]; then
    # 获取 launcher_ip
    launcher_ip=$(head -n 1 $pssh_hosts_path)

    # 生成 hostfile_{launcher_ip}
    HOSTFILE_DIR=$work_dir/"hostfiles"
    if [ ! -d "$HOSTFILE_DIR" ]; then
        mkdir -p "$HOSTFILE_DIR"
    fi
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
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=${GRADIENT_ACC:-1}

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
OUTPUT_DIR="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/Qwen2-VL-Finetune/output/deplot_mind_pure_1031_27w_no_gpt_yewu"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
# ==== 选择模型 ==== #
# MODEL_NAME="/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-2B-Instruct"
MODEL_NAME="/apdcephfs_sh8/share_301266059/stephenruan/code/src/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

echo $NUM_NODES
echo $MASTER_ADDR
echo $GPUS
echo $MASTER_PORT

# ==== 启动训练 ==== #
deepspeed \
    --hostfile $HOSTFILE \
    --num_nodes=$NUM_NODES \
    --master_addr=$MASTER_ADDR \
    --num_gpus=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/training/train.py \
    --model_id $MODEL_NAME \
    --data_path /apdcephfs_sh8/share_301266059/stephenruan/data/deplot_yewu/deplot_mind_pure_1031_27w_no_gpt_yewu.jsonl \
    --max_seq_length 8192 \
    --freeze_vision_tower True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --min_pixels $((1 * 28 * 28)) \
    --max_pixels $((3072 * 28 * 28)) \
    --learning_rate 2e-5 \
    --merger_lr 2e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --save_strategy "epoch" \
    --save_steps 200 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --disable_flash_attn2 False \
    --deepspeed scripts/zero_stage1_config.json \
    --report_to tensorboard \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
  
# ==== 训练中断或结束后挂起 ==== #
cd /apdcephfs_sh8/share_301266059/stephenruan/code/hang
bash occupy.sh