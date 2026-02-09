#!/bin/bash
set -x

CUR_DIR=`pwd`
export PYTHONPATH="${PYTHONPATH}:${CUR_DIR}"
export TF_ENABLE_ONEDNN_OPTS=0
export BABBLE_VERB=0


if [[ -n "${KUBEBRAIN_REPLICA}" && -n "${KUBEBRAIN_REPLICA_TOTAL}" ]]; then
    echo "[ARGS] KUBEBRAIN_REPLICA: ${KUBEBRAIN_REPLICA}"
    echo "[ARGS] KUBEBRAIN_REPLICA_TOTAL: ${KUBEBRAIN_REPLICA_TOTAL}"
    rank=${KUBEBRAIN_REPLICA}
    world_size=${KUBEBRAIN_REPLICA_TOTAL}
elif [[ -n "${RLAUNCH_REPLICA}" && -n "${RLAUNCH_REPLICA_TOTAL}" ]]; then
    echo "[ARGS] RLAUNCH_REPLICA: ${RLAUNCH_REPLICA}"
    echo "[ARGS] RLAUNCH_REPLICA_TOTAL: ${RLAUNCH_REPLICA_TOTAL}"
    rank=${RLAUNCH_REPLICA}
    world_size=${RLAUNCH_REPLICA_TOTAL}
else
    rank=0
    world_size=1
fi

echo "[ARGS] world_size: ${world_size}"
echo "[ARGS] rank: ${rank}"

workers=8
step=$((workers-1))
for i in $(seq 0 $step); do
    gpu_id=$i
    new_rank=$((rank * ${workers} + i))
    new_world_size=$((world_size * ${workers}))
    echo "[ARGS] new_rank: ${new_rank} new_world_size: ${new_world_size}"
    echo "Started process on GPU ${gpu_id}"
    export CUDA_VISIBLE_DEVICES=${gpu_id} && \
        python3 ./wavlm_feature.py -w $new_world_size -r $new_rank &
done

# 等待所有后台进程完成
wait
echo "All processes completed"
# rjob submit --name preparewebdataset \
#     --replica=8 \
#     --custom-resources rdma/mlnx_shared=8 \
#     --charged-group=tts \ 
#     --private-machine=group \
#     --preemptible=no \
#     --mount=juicefs+s3://oss.i.shaipower.com/fcl-jfs:/mnt/fcl-jfs \
#     --cpu=112 --gpu=16 --memory=819200 \
#     --host-network=true \
#     --image hub.i.basemind.com/step-music/music-token-server-fcl:python3.10-cuda11.8-torchaudio2.4-stepbps-20251215-115411-481774055-v36shc \
#     -- bash -c "pip3 install matplotlib==3.10.1 rotary-embedding-torch==0.8.6 beartype==0.21.0 ml_collections loralib transformers==4.57.3 funasr==1.2.7 jiwer==3.1.0 langdetect==1.0.9 && cd /mnt/fcl-jfs/mss_tool && bash ./mss_asr_filter.sh"


brainctl launch  \
    --charged-group=tts \
    --custom-resources rdma/mlnx_shared=8 \
    --private-machine=group \
    --preemptible=no \
    --mount=juicefs+s3://oss.i.shaipower.com/fcl-jfs:/mnt/fcl-jfs \
    --mount=juicefs+s3://oss.i.shaipower.com/yi-music:/mnt/yi-jfs \
    --cpu=100 --gpu=8 --memory=500000 --positive-tags L40S,feature/gpfs=yes \
    -- bash 
