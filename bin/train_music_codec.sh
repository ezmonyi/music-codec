#!/bin/bash
set -eo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH

# Example (8 GPU): ./bin/train_music_codec.sh conf/single_vq.yaml my_exp
# Example RVQ:    ./bin/train_music_codec.sh conf/rvq_8x1024.yaml my_exp
# Override: model_dir=/path tensorboard_dir=/path ./bin/train_music_codec.sh conf/single_vq.yaml my_exp
config_yaml="${1:-conf/single_vq.yaml}"
exp_name="${2:-exp}"

# Resolve config path and conf_name (basename without .yaml) for dirs
conf="${config_yaml}"
conf_name="${conf_name:-$(basename "${config_yaml}" .yaml)}"

# Default paths (8-GPU machine)
model_dir="${model_dir:-/mnt/yi-jfs/checkpoints/codec/${conf_name}_${exp_name}}"
tensorboard_dir="${tensorboard_dir:-/mnt/yi-jfs/tensorboard/codec_${conf_name}_${exp_name}}"
dataset_conf="${dataset_conf:-conf/dataset_codec.yaml}"
restore_model_path="${restore_model_path:-}"

# DDP (default 8 GPU per node)
num_nodes="${num_nodes:-1}"
node_idx="${node_idx:-0}"
gpus_per_node="${gpus_per_node:-8}"
master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-29500}"

# Data / training
batch_size="${batch_size:-4}"
num_workers="${num_workers:-8}"
prefetch="${prefetch:-4}"
timeout="${timeout:-300}"
pin_memory="${pin_memory:-true}"

# Parse --option value overrides (skip first two positional: conf_name, exp_name)
shift 2 2>/dev/null || true
if [ -f "bin/parse_options.sh" ]; then
  source bin/parse_options.sh "$@"
fi

echo "[ARGS] conf=${conf}"
echo "[ARGS] dataset_conf=${dataset_conf}"
echo "[ARGS] model_dir=${model_dir}"
echo "[ARGS] tensorboard_dir=${tensorboard_dir}"
echo "[ARGS] num_nodes=${num_nodes} node_idx=${node_idx} gpus_per_node=${gpus_per_node}"
echo "[ARGS] batch_size=${batch_size} num_workers=${num_workers}"

export WORLD_SIZE=$((num_nodes * gpus_per_node))
mkdir -p "${model_dir}"
mkdir -p "${tensorboard_dir}"

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
[ -n "${restore_model_path}" ] && _train_args+=(--restore_model_path "${restore_model_path}")
[ "${pin_memory}" = "true" ] && _train_args+=(--pin_memory)

python -m torch.distributed.launch \
  --master_addr="${master_addr}" \
  --master_port="${master_port}" \
  --nproc_per_node="${gpus_per_node}" \
  --nnodes="${num_nodes}" \
  --node_rank="${node_idx}" \
  bin/train.py "${_train_args[@]}"
