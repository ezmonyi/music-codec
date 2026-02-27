#!/bin/bash
# Launch DDP training on multiple 8-GPU machines via SSH.
#
# Usage:
#   NODES="node0,node1,node2" MASTER_ADDR=192.168.1.1 ./bin/run_multinode.sh conf/single_vq.yaml [--batch_size 8 ...]
#
# Env:
#   NODES          Comma-separated list of hostnames/IPs (node0 = master)
#   MASTER_ADDR    IP or hostname of master (node0). Required for multi-node.
#   MASTER_PORT    Port for rendezvous (default 29500)
#   WORK_DIR       Remote working directory (default: current dir on master)
#   SSH_USER       SSH user (default: $USER)
#   SSH_OPTS       Extra SSH options
#
# Example (2×8 GPU):
#   NODES="gpu01,gpu02" MASTER_ADDR=gpu01 ./bin/run_multinode.sh conf/single_vq.yaml
#
set -eo pipefail

config_yaml="${1:-}"
[ -z "${config_yaml}" ] && echo "Usage: NODES=host1,host2 MASTER_ADDR=host1 $0 conf/config.yaml [--option value ...]" && exit 1
shift || true

nodes_str="${NODES:-}"
master_addr="${MASTER_ADDR:-}"
master_port="${MASTER_PORT:-29500}"
work_dir="${WORK_DIR:-}"
ssh_user="${SSH_USER:-$USER}"
ssh_opts="${SSH_OPTS:-}"

[ -z "${nodes_str}" ] && echo "Error: NODES is required (e.g. NODES=node0,node1)" && exit 1
[ -z "${master_addr}" ] && echo "Error: MASTER_ADDR is required (IP/hostname of first node)" && exit 1

IFS=',' read -ra NODE_LIST <<< "$nodes_str"
num_nodes=${#NODE_LIST[@]}
[ "$num_nodes" -lt 2 ] && echo "Error: NODES should have at least 2 hosts for multi-node" && exit 1

# work_dir: must be the same path on all nodes (e.g. NFS mount)
[ -z "${work_dir}" ] && work_dir="$(pwd)" && echo "[MULTINODE] work_dir not set, using current dir: ${work_dir}"

# Build extra args to pass through
extra_args=()
for arg in "$@"; do extra_args+=("$arg"); done

# Export rdzv_id so all nodes use the same job id
export rdzv_id="codec_multinode_$$_$(date +%s)"

echo "[MULTINODE] Launching on ${num_nodes} nodes: ${nodes_str}"
echo "[MULTINODE] master_addr=${master_addr} master_port=${master_port} work_dir=${work_dir}"
echo "[MULTINODE] rdzv_id=${rdzv_id}"

# Build remote command with properly escaped extra args
_escaped_args=()
for arg in "${extra_args[@]}"; do _escaped_args+=("$(printf '%q' "$arg")"); done

# Launch on each node in parallel
for i in "${!NODE_LIST[@]}"; do
  node="${NODE_LIST[$i]}"
  node_rank=$i
  ssh_target="${ssh_user}@${node}"
  remote_cmd="cd $(printf '%q' "${work_dir}") && num_nodes=${num_nodes} node_idx=${node_rank} master_addr=${master_addr} master_port=${master_port} rdzv_id=${rdzv_id} ./bin/train_music_codec.sh $(printf '%q' "${config_yaml}") ${_escaped_args[*]}"
  echo "[MULTINODE] Node ${node_rank}: ${node}"
  ssh ${ssh_opts} -f "${ssh_target}" "${remote_cmd}" &
done
wait
echo "[MULTINODE] All nodes started. Check logs on each node."
