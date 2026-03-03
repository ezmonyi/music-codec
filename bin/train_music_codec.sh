#!/bin/bash
set -eo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH
export TF_ENABLE_ONEDNN_OPTS=0
export BABBLE_VERB=0
export USE_NCCL=1
export USE_SYSTEM_NCCL=1 
export NCCL_DEBUG=WARN # INFO
export NCCL_IB_GID_INDEX=5
export NCCL_IB_HCA=mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TC=186
export NCCL_IB_DISABLE=0
export NCCL_PXN_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=18
export NCCL_IB_RETRY_CNT=7
# Multi-node: if nodes have different IB HCAs, set NCCL_IB_DISABLE=1 (TCP fallback) before running
export OMP_NUM_THREADS=16

# Example (8 GPU single node): ./bin/train_music_codec.sh conf/single_vq.yaml
# Example:                     ./bin/train_music_codec.sh conf/single_vq_enable_schedule_disable_ema_disc.yaml
# Override:                    model_dir=/path ./bin/train_music_codec.sh conf/single_vq.yaml
#
# Multi-node (e.g. 2×8 GPU): Run the SAME command on each machine with different node_idx:
#   Node 0 (master): ./bin/train_music_codec.sh conf/single_vq.yaml --num_nodes 2 --node_idx 0 --master_addr <MASTER_IP>
#   Node 1 (worker): ./bin/train_music_codec.sh conf/single_vq.yaml --num_nodes 2 --node_idx 1 --master_addr <MASTER_IP>
#   Use run_multinode.sh to orchestrate via SSH, or run manually on each node.
config_yaml="${1:-conf/single_vq.yaml}"
exp_name="${exp_name:-$(date +"%y%m%d_%H%M")}"
# Collapse any double (or more) underscores so dir names stay clean (e.g. 260224__143 -> 260224_143)
while [[ "${exp_name}" == *"__"* ]]; do exp_name="${exp_name//__/_}"; done

# Resolve config path and conf_name (basename without .yaml) for dirs
conf="${config_yaml}"
conf_name="${conf_name:-$(basename "${config_yaml}" .yaml)}"

# Default paths (8-GPU machine)
model_dir="${model_dir:-/mnt/yi-jfs/checkpoints/codec/${conf_name}_${exp_name}}"
tensorboard_dir="${tensorboard_dir:-/mnt/yi-jfs/tensorboard/codec_${conf_name}_${exp_name}}"
dataset_conf="${dataset_conf:-conf/dataset_codec.yaml}"

# DDP (default 8 GPU per node; multi-node: set num_nodes, node_idx, master_addr)
num_nodes="${num_nodes:-1}"
node_idx="${node_idx:-0}"
gpus_per_node="${gpus_per_node:-8}"
master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-29500}"
rdzv_id="${rdzv_id:-}"  # Unique job id for multi-node rendezvous (auto-set if empty)

# Data / training (conservative defaults to avoid pod OOM; override if you have more RAM)
batch_size="${batch_size:-4}"
num_workers="${num_workers:-2}"
prefetch="${prefetch:-2}"
timeout="${timeout:-300}"
pin_memory="${pin_memory:-true}"

# Parse --option value overrides (skip first positional: config_yaml)
shift 1 2>/dev/null || true
if [ -f "bin/parse_options.sh" ]; then
  source bin/parse_options.sh "$@"
fi

echo "[ARGS] conf=${conf} exp_name=${exp_name}"
echo "[ARGS] dataset_conf=${dataset_conf}"
echo "[ARGS] model_dir=${model_dir}"
echo "[ARGS] tensorboard_dir=${tensorboard_dir}"
echo "[ARGS] num_nodes=${num_nodes} node_idx=${node_idx} gpus_per_node=${gpus_per_node}"
echo "[ARGS] batch_size=${batch_size} num_workers=${num_workers}"

export WORLD_SIZE=$((num_nodes * gpus_per_node))
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"
[ -z "${rdzv_id}" ] && rdzv_id="codec_${conf_name}_${exp_name}_$$"
mkdir -p "${model_dir}"
mkdir -p "${tensorboard_dir}"

# All logs under dedicated dir (multi-node: per-node log to avoid interleaving)
log_dir="${log_dir:-/mnt/yi-jfs/logs/codec}"
mkdir -p "${log_dir}"
timestamp=$(date +"%Y%m%d_%H%M%S")
if [ "${num_nodes}" -gt 1 ]; then
  log_file="${log_dir}/train_${conf_name}_${exp_name}_${timestamp}_node${node_idx}.log"
else
  log_file="${log_dir}/train_${conf_name}_${exp_name}_${timestamp}.log"
fi
echo "[ARGS] log_file=${log_file}"

# Model cache: HF models → /data/.huggingface; Whisper → dataset conf whisper_download_root
export HF_HOME="${HF_HOME:-/data/.huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"

# Optional NCCL / env (tune for your 8-GPU machine if needed)
export TF_ENABLE_ONEDNN_OPTS=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

_train_args=(
  --config "${conf}"
  --dataset_conf "${dataset_conf}"
  --model_dir "${model_dir}"
  --tensorboard_dir "${tensorboard_dir}"
  --batch_size "${batch_size}"
  --num_workers "${num_workers}"
  --prefetch "${prefetch}"
  --timeout "${timeout}"
)
# restore_model_path: set in config yaml (train_conf.restore_model_path) to load a ckpt e.g. after run_vq_init
[ "${pin_memory}" = "true" ] && _train_args+=(--pin_memory)

# Run training: use torchrun (recommended) or fallback to torch.distributed.launch
# torchrun: --rdzv-endpoint replaces master_addr+port; same env vars (MASTER_ADDR/PORT) are set for children
if command -v torchrun &>/dev/null; then
  _launcher="torchrun"
  _launcher_args=(
    --rdzv-backend=c10d
    --rdzv-endpoint="${master_addr}:${master_port}"
    --rdzv-id="${rdzv_id}"
    --nnodes="${num_nodes}"
    --node-rank="${node_idx}"
    --nproc-per-node="${gpus_per_node}"
  )
else
  _launcher="python -m torch.distributed.launch"
  _launcher_args=(
    --master_addr="${master_addr}"
    --master_port="${master_port}"
    --nproc_per_node="${gpus_per_node}"
    --nnodes="${num_nodes}"
    --node_rank="${node_idx}"
  )
fi
echo "[LAUNCH] Using ${_launcher} (num_nodes=${num_nodes} world_size=${WORLD_SIZE})"
# Note: ${_launcher} unquoted so "python -m torch.distributed.launch" word-splits correctly
${_launcher} "${_launcher_args[@]}" bin/train.py "${_train_args[@]}" 2>&1 | tee "${log_file}"
