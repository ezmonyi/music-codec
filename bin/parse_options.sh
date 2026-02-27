#!/usr/bin/env bash
# Parse --option value and set shell variable option_name=value.
# Source with: source bin/parse_options.sh
#
# Multi-node DDP options (must be defined in caller before sourcing):
#   --num_nodes, --node_idx, --master_addr, --master_port, --rdzv_id,
#   --gpus_per_node, --batch_size, --num_workers, etc.

for ((argpos=1; argpos<$#; argpos++)); do
  if [ "${!argpos}" == "--config" ]; then
    argpos_plus1=$((argpos+1))
    config=${!argpos_plus1}
    [ ! -r "$config" ] && echo "$0: missing config '$config'" && exit 1
    . "$config"
  fi
done

while true; do
  [ -z "${1:-}" ] && break
  case "$1" in
    --*=*) echo "$0: use --name value form, got '$1'"; exit 1 ;;
    --*)
      name=$(echo "$1" | sed s/^--// | sed s/-/_/g)
      eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1
      eval $name=\"$2\"
      shift 2
      ;;
    *) break ;;
  esac
done
true
