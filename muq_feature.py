import argparse
import io
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from multiprocessing import Process

import numpy as np
import torch
import torchaudio
from MuQ import MuQ
from oss_cli import OSSPool
from tqdm import tqdm

ossb_pool = OSSPool(datatype="B")

def load_audio_torch(audio_path, oss_client=None, target_sample_rate=None):
    if target_sample_rate is None:
        target_sample_rate = 24000

    if os.path.exists(audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if waveform.shape[0] > 1:
                waveform = waveform[0, :]
                waveform = waveform.unsqueeze(0)

            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
            return waveform, True, ''

        except Exception as e:
            print(f"Error in loading audio: {audio_path}, {e}")
            traceback.print_exc()
            return None, False, str(e)
        
    elif audio_path.startswith("s3://"):
        try:
            bucket_name = audio_path[5:].split("/", 1)[0]
            filename = audio_path[5:].split("/", 1)[1]
            if oss_client is None:
                oss_client = ossb_pool.get_conn()
            else:
                oss_client = oss_client
            
            wavdata = oss_client.get_file(bucket_name, filename).read()
            audio_io = io.BytesIO(wavdata)
            waveform, sample_rate = torchaudio.load(audio_io)

            if waveform.shape[0] > 1:
                waveform = waveform[0, :]
                waveform = waveform.unsqueeze(0)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
            return waveform, True, ''
        
        except Exception as e:
            print(f"Error in loading audio: {audio_path}, {e}")
            return None, False, str(e)
    else:
        return None, False, "audio_path not found"

def save_to_file(results, json_file):
    valid_results = [r for r in results if r is not None]
    with open(json_file, "a") as f:
        f.writelines(valid_results)


def process_muq_single_line(line, oss_client, model, device, output_dir, rank, feat_index):
    """单条文件路径：加载 24kHz 音频，用 MuQ 提取特征，保存 last_hidden_state 到 .npy，返回带路径的 JSON 行。"""
    try:
        audio_path = line
        signals, ready, err_msg = load_audio_torch(
            audio_path, oss_client=oss_client, target_sample_rate=24000
        )
        if signals is None:
            print(f"audio_path: {audio_path} load failed")
            return None
        dur = round(signals.shape[1] / 24000, 2)
        with torch.no_grad():
            wavs = signals.to(device)
            output = model(wavs, output_hidden_states=True)
            # last_hidden_state: [B, T, D]
            feats = output.last_hidden_state.cpu()
        feats_np = feats.numpy()
        feats_dir = os.path.join(output_dir, "muq_feats")
        os.makedirs(feats_dir, exist_ok=True)
        feat_path = os.path.join(feats_dir, f"feat_{rank}_{feat_index}.npy")
        np.save(feat_path, feats_np)
        jsd = {
            "audio_filepath": audio_path,
            "muq_feature_path": feat_path,
            "muq_feature_shape": list(feats_np.shape),
            "duration": dur
        }
        return json.dumps(jsd, ensure_ascii=False) + "\n"
    except Exception as e:
        print(f"process_muq_single_line error: {e}")
        return None


def process_features(lines, save_file, output_dir, device, rank=0, model_name="OpenMuQ/MuQ-large-msd-iter"):
    """批量用 MuQ 提取特征：加载模型一次，逐条处理并写入 save_file，特征存 output_dir/muq_feats/*.npy。"""
    muq = MuQ.from_pretrained(model_name)
    muq = muq.to(device).eval()
    print("MuQ model loaded for feature extraction.")
    results = []
    for idx, line in enumerate(tqdm(lines)):
        oss_client = ossb_pool.get_conn()
        ret = process_muq_single_line(line, oss_client, muq, device, output_dir, rank, idx)
        ossb_pool.release_conn(oss_client)
        if ret is not None:
            results.append(ret)
        if len(results) >= 500:
            save_to_file(results, save_file)
            results = []
    if len(results) > 0:
        save_to_file(results, save_file)


def split_data(lines, world_size, rank):
    chunk_size = len(lines) // world_size
    print(f"chunk_size: {chunk_size}")
    rest = len(lines) - world_size * chunk_size
    all_parts = [chunk_size] * world_size
    for i in range(rest):
        all_parts[i] += 1
    assert sum(all_parts) == len(lines)
    cut_lines = lines[sum(all_parts[:rank]): sum(all_parts[:rank+1])]
    return cut_lines


def read_file_efficiently(input_jsonfile, count=None, world_size=1, rank=0):
    """高效读取大文件，按需分配内存"""
    total_lines = 0
    target_start = 0
    target_end = 0
    
    # 第一遍：计算总行数
    print(f"Counting total lines in {input_jsonfile}...")
    with open(input_jsonfile, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
    
    if count is not None:
        total_lines = min(total_lines, count)
    
    # 计算当前rank需要处理的行范围
    chunk_size = total_lines // world_size
    rest = total_lines - world_size * chunk_size
    all_parts = [chunk_size] * world_size
    for i in range(rest):
        all_parts[i] += 1
    
    target_start = sum(all_parts[:rank])
    target_end = sum(all_parts[:rank+1])
    
    print(f"rank {rank}: processing lines {target_start} to {target_end-1} (total: {total_lines})")
    
    # 第二遍：只读取需要的行
    current_line = 0
    target_lines = []
    
    with open(input_jsonfile, "r", encoding="utf-8") as f:
        for line in f:
            if current_line >= target_start and current_line < target_end:
                target_lines.append(line)
            elif current_line >= target_end:
                break
            current_line += 1
    
    return target_lines



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_jsonfile", type=str, default="/mnt/fcl-jfs/music_tokenizer/codec/qq_music_path_with_metainfo.jsonl")
    parser.add_argument("-o", "--output_dir", type=str, default="/mnt/fcl-jfs/music_tokenizer/codec/qq_music_sky_code_with_prompt")
    parser.add_argument("-n", "--count", type=int, default=None)
    parser.add_argument("-w", "--world_size", type=int, default=1)
    parser.add_argument("-r", "--rank", type=int, default=0)
    
    args = parser.parse_args()
    count = args.count
    input_jsonfile = args.input_jsonfile
    output_dir = args.output_dir
    
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')  # 避免fork带来的CUDA上下文问题

    replica = 1
    rank = 0 if args.rank < 0 else args.rank
    world_size = replica if args.world_size < 0 else args.world_size
    print(f"rank: {rank}, world_size: {world_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    cut_lines = read_file_efficiently(input_jsonfile, count, world_size, rank)
    print(f"cut_lines: {len(cut_lines)}")

    os.makedirs(output_dir, exist_ok=True)
    process_muq_feature(
        cut_lines,
        save_file=f"{output_dir}/music_muq_{world_size}_{rank}.jsonl",
        output_dir=output_dir,
        device=device,
        rank=rank,
    )
    print(f"all process done")