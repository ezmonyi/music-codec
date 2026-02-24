#!/bin/bash
# K-means VQ codebook init on 8 GPUs (single machine). Same dataset/config as training; logs under /mnt/yi-jfs/logs/codec.
# Usage: ./bin/run_vq_init_8gpu.sh <config_yaml> <out_ckpt>
#        ./bin/run_vq_init_8gpu.sh <config_yaml> <dataset_conf> <out_ckpt>
#   dataset_conf defaults to conf/dataset_codec.yaml (same as train_music_codec.sh)
#   e.g. ./bin/run_vq_init_8gpu.sh conf/single_vq.yaml /mnt/yi-jfs/checkpoints/codec/vq_init/model_0.pt

set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}:$PWD"

config_yaml="${1:-conf/single_vq.yaml}"
# Same default dataset config as train_music_codec.sh
dataset_conf="${dataset_conf:-conf/dataset_codec.yaml}"
if [ "$#" -ge 3 ]; then
  dataset_conf="${2}"
  out_ckpt="${3}"
else
  out_ckpt="${2:?out_ckpt required (e.g. /mnt/yi-jfs/checkpoints/codec/vq_init/model_0.pt)}"
fi

max_samples="${max_samples:-1000000}"
max_vectors="${max_vectors:-10000000}"
kmeans_method="${kmeans_method:-minibatch}"  # minibatch (faster) or full (slower, more accurate)
kmeans_max_iter="${kmeans_max_iter:-100}"
kmeans_n_init="${kmeans_n_init:-1}"
kmeans_batch_size="${kmeans_batch_size:-50000}"
gpus_per_node="${gpus_per_node:-8}"

log_dir="${log_dir:-/mnt/yi-jfs/logs/codec}"
mkdir -p "${log_dir}"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/vq_init_8gpu_${timestamp}.log"

echo "[ARGS] config=${config_yaml} dataset_conf=${dataset_conf} out=${out_ckpt}"
echo "[ARGS] max_samples=${max_samples} max_vectors=${max_vectors} gpus=${gpus_per_node}"
echo "[ARGS] kmeans_method=${kmeans_method} max_iter=${kmeans_max_iter} n_init=${kmeans_n_init}"
echo "[ARGS] log_file=${log_file}"

torchrun --nproc_per_node="${gpus_per_node}" bin/run_vq_init.py \
  --config "${config_yaml}" \
  --dataset_conf "${dataset_conf}" \
  --out "${out_ckpt}" \
  --max_samples "${max_samples}" \
  --max_vectors "${max_vectors}" \
  --kmeans_method "${kmeans_method}" \
  --kmeans_max_iter "${kmeans_max_iter}" \
  --kmeans_n_init "${kmeans_n_init}" \
  --kmeans_batch_size "${kmeans_batch_size}" \
  2>&1 | tee "${log_file}"
