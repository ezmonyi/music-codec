#!/bin/bash
set -eo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH

# Use one Python for deps and for launching (avoids torchrun from another env missing hyperpyyaml)
RUN_PYTHON="${RUN_PYTHON:-$(command -v python 2>/dev/null || command -v python3)}"
"$RUN_PYTHON" -c "import hyperpyyaml" 2>/dev/null || "$RUN_PYTHON" -m pip install -q hyperpyyaml

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

# Example (single node):  ./bin/train.sh conf/mel_single_vq.yaml my_exp
# Ablation (standard):    ./bin/train.sh conf/mel_fm_only.yaml ablation_llama --estimator llama
#                         ./bin/train.sh conf/mel_fm_only.yaml ablation_dit   --estimator dit
# EAR_VAE latent:         ./bin/train.sh conf/earvae_fm_only.yaml earvae_exp
# Override:               model_dir=/path ./bin/train.sh conf/mel_single_vq.yaml my_exp
#
# Multi-node (e.g. 2×8 GPU): Run the SAME command on each machine with different node_idx:
#   Node 0 (master): ./bin/train.sh conf/mel_single_vq.yaml my_exp --num_nodes 2 --node_idx 0 --master_addr <MASTER_IP>
#   Node 1 (worker): ./bin/train.sh conf/mel_single_vq.yaml my_exp --num_nodes 2 --node_idx 1 --master_addr <MASTER_IP>
#   Use run_multinode.sh to orchestrate via SSH, or run manually on each node.
config_yaml="${1:-conf/mel_single_vq.yaml}"
exp_name_arg="${2:-}"
# Priority: env exp_name > positional arg > timestamp fallback
if [ -n "${exp_name}" ]; then
  :  # keep existing env exp_name
elif [ -n "${exp_name_arg}" ]; then
  exp_name="${exp_name_arg}"
else
  exp_name="$(date +"%y%m%d_%H%M")"
fi
# Collapse any double (or more) underscores so dir names stay clean (e.g. 260224__143 -> 260224_143)
while [[ "${exp_name}" == *"__"* ]]; do exp_name="${exp_name//__/_}"; done

# Resolve config path and conf_name (basename without .yaml) for dirs
conf="${config_yaml}"
conf_name="${conf_name:-$(basename "${config_yaml}" .yaml)}"

# Default paths (8-GPU machine)
model_dir="${model_dir:-/mnt/yi-jfs/checkpoints/codec/${conf_name}_${exp_name}}"
tensorboard_dir="${tensorboard_dir:-/mnt/yi-jfs/tensorboard/codec_${conf_name}_${exp_name}}"
dataset_conf="${dataset_conf:-}"  # dataset config now embedded in training yaml; set to override

# DDP (default: auto-detect GPUs per node; multi-node: set num_nodes, node_idx, master_addr)
num_nodes="${num_nodes:-1}"
node_idx="${node_idx:-0}"

# Auto-detect GPU count if gpus_per_node is not set (fallback to 1 if nvidia-smi is unavailable)
gpu_auto_detected=0
if [ -z "${gpus_per_node:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    gpus_per_node=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${gpus_per_node:-0}" -lt 1 ]; then
      echo "[WARN] nvidia-smi -L returned 0 GPUs; using gpus_per_node=1 (set gpus_per_node explicitly if wrong)"
      gpus_per_node=1
    fi
  else
    gpus_per_node=1
  fi
  gpu_auto_detected=1
fi
gpus_per_node="${gpus_per_node:-1}"
if [ "${gpus_per_node}" -lt 1 ]; then
  echo "[WARN] gpus_per_node must be >= 1; overriding to 1"
  gpus_per_node=1
fi

master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-29500}"
rdzv_id="${rdzv_id:-}"  # Unique job id for multi-node rendezvous (auto-set if empty)

# Data / training
# Note:
#   - batch_size: leave empty by default so train_conf.batch_size in YAML takes effect.
#                 Set env var batch_size or pass --batch_size to override explicitly.
#   - num_workers/prefetch/timeout/pin_memory: runtime defaults for this script.
batch_size="${batch_size:-}"
estimator="${estimator:-}"
overrides="${overrides:-}"
num_workers="${num_workers:-2}"
prefetch="${prefetch:-2}"
timeout="${timeout:-300}"
pin_memory="${pin_memory:-true}"

# Parse --option value overrides (skip first two positionals: config_yaml, exp_name)
shift 2 2>/dev/null || true
if [ -f "bin/parse_options.sh" ]; then
  source bin/parse_options.sh "$@"
fi

# Shorthand: estimator=dit|llama → append to overrides
if [ -n "${estimator}" ]; then
  overrides="${overrides:+${overrides}
}estimator_type: ${estimator}"
fi

echo "[ARGS] conf=${conf} exp_name=${exp_name}"
echo "[ARGS] dataset_conf=${dataset_conf}"
echo "[ARGS] model_dir=${model_dir}"
echo "[ARGS] tensorboard_dir=${tensorboard_dir}"
if [ "${gpu_auto_detected}" -eq 1 ]; then
  echo "[ARGS] num_nodes=${num_nodes} node_idx=${node_idx} gpus_per_node=${gpus_per_node} (auto-detected via nvidia-smi)"
else
  echo "[ARGS] num_nodes=${num_nodes} node_idx=${node_idx} gpus_per_node=${gpus_per_node}"
fi
echo "[ARGS] batch_size=${batch_size:-\"<from_yaml>\"} num_workers=${num_workers}"
[ -n "${overrides}" ] && echo "[ARGS] overrides=${overrides}"

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
  --model_dir "${model_dir}"
  --tensorboard_dir "${tensorboard_dir}"
  --num_workers "${num_workers}"
  --prefetch "${prefetch}"
  --timeout "${timeout}"
)
# Only pass --dataset_conf when explicitly set (dataset config is now in training yaml)
if [ -n "${dataset_conf}" ]; then
  _train_args+=(--dataset_conf "${dataset_conf}")
fi
# Only override YAML batch_size when explicitly set
if [ -n "${batch_size}" ]; then
  _train_args+=(--batch_size "${batch_size}")
fi
# restore_model_path: set in config yaml (train_conf.restore_model_path) to load a ckpt e.g. after run_vq_init
[ "${pin_memory}" = "true" ] && _train_args+=(--pin_memory)
if [ -n "${overrides}" ]; then
  _train_args+=(--overrides "${overrides}")
fi

# Run training with the same Python that has hyperpyyaml (avoid PATH torchrun from another env)
# Prefer torch.distributed.run (PyTorch 1.9+), else torch.distributed.launch
if "$RUN_PYTHON" -m torch.distributed.run --help &>/dev/null; then
  _launcher=("$RUN_PYTHON" -m torch.distributed.run
    --rdzv-backend=c10d
    --rdzv-endpoint="${master_addr}:${master_port}"
    --rdzv-id="${rdzv_id}"
    --nnodes="${num_nodes}"
    --node-rank="${node_idx}"
    --nproc-per-node="${gpus_per_node}"
  )
else
  _launcher=("$RUN_PYTHON" -m torch.distributed.launch
    --master_addr="${master_addr}"
    --master_port="${master_port}"
    --nproc_per_node="${gpus_per_node}"
    --nnodes="${num_nodes}"
    --node_rank="${node_idx}"
  )
fi
echo "[LAUNCH] Using $RUN_PYTHON -m torch.distributed.* (num_nodes=${num_nodes} world_size=${WORLD_SIZE})"
"${_launcher[@]}" bin/train.py "${_train_args[@]}" 2>&1 | tee "${log_file}"
