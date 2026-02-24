#!/bin/bash
# FM-only training on 8 GPUs: encoder + CFM without VQ.
# Same dataset/NCCL config as train_music_codec.sh.
#
# Usage:
#   ./bin/run_fm_only.sh [conf/fm_only.yaml]
#   batch_size=8 ./bin/run_fm_only.sh conf/fm_only.yaml
#
# After training, use the exported checkpoint in main training:
#   Set restore_model_path in single_vq.yaml to the encoder_cfm_*.pt path.

set -eo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH
export TF_ENABLE_ONEDNN_OPTS=0
export BABBLE_VERB=0
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export NCCL_DEBUG=WARN
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
export OMP_NUM_THREADS=16

config_yaml="${1:-conf/fm_only.yaml}"
exp_name="${exp_name:-$(date +"%y%m%d_%H%M")}"

conf="${config_yaml}"
conf_name="${conf_name:-$(basename "${config_yaml}" .yaml)}"

model_dir="${model_dir:-/mnt/yi-jfs/checkpoints/codec/fm_only_${conf_name}_${exp_name}}"
tensorboard_dir="${tensorboard_dir:-/mnt/yi-jfs/tensorboard/fm_only_${conf_name}_${exp_name}}"
dataset_conf="${dataset_conf:-conf/dataset_codec.yaml}"

# DDP
num_nodes="${num_nodes:-1}"
node_idx="${node_idx:-0}"
gpus_per_node="${gpus_per_node:-8}"
master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-29500}"

# Data / training
batch_size="${batch_size:-12}"
num_workers="${num_workers:-8}"
prefetch="${prefetch:-4}"
timeout="${timeout:-300}"
pin_memory="${pin_memory:-true}"

# Parse --option value overrides
shift 1 2>/dev/null || true
if [ -f "bin/parse_options.sh" ]; then
  source bin/parse_options.sh "$@"
fi

echo "[FM-ONLY] conf=${conf} exp_name=${exp_name}"
echo "[FM-ONLY] dataset_conf=${dataset_conf}"
echo "[FM-ONLY] model_dir=${model_dir}"
echo "[FM-ONLY] tensorboard_dir=${tensorboard_dir}"
echo "[FM-ONLY] num_nodes=${num_nodes} node_idx=${node_idx} gpus_per_node=${gpus_per_node}"
echo "[FM-ONLY] batch_size=${batch_size} num_workers=${num_workers}"

export WORLD_SIZE=$((num_nodes * gpus_per_node))
mkdir -p "${model_dir}"
mkdir -p "${tensorboard_dir}"

log_dir="${log_dir:-/mnt/yi-jfs/logs/codec}"
mkdir -p "${log_dir}"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/fm_only_${conf_name}_${exp_name}_${timestamp}.log"
echo "[FM-ONLY] log_file=${log_file}"

export HF_HOME="${HF_HOME:-/data/.huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
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
[ "${pin_memory}" = "true" ] && _train_args+=(--pin_memory)

python -m torch.distributed.launch \
  --master_addr="${master_addr}" \
  --master_port="${master_port}" \
  --nproc_per_node="${gpus_per_node}" \
  --nnodes="${num_nodes}" \
  --node_rank="${node_idx}" \
  bin/train_fm_only.py "${_train_args[@]}" 2>&1 | tee "${log_file}"
