import io
import json
import os
import queue
import tarfile
import threading
import time
from subprocess import CalledProcessError, run

import numpy as np
import torch
import torch.multiprocessing as mp
from muq import MuQ
from oss_cli import OSSPool
from WavLM import WavLM, WavLMConfig
from whisper import whisper
from whisper.whisper.audio import (
    N_FRAMES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

WHISPER_CKPT = "/mnt/yi-jfs/pretrained_models/whisper/large-v3.pt"
MUQ_PRETRAINED = "OpenMuQ/MuQ-large-msd-iter"
WAVLM_CKPT = "/mnt/yi-jfs/pretrained_models/wavlm/WavLM-Large.pt"

NUM_GPUS = 8
PREFETCH_DEPTH = 8
NUM_PREFETCH_THREADS = 24


# ---------------------------------------------------------------------------
# Utility: audio decoding
# ---------------------------------------------------------------------------

def audio_bytes_resample(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le", "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr), "-",
    ]
    try:
        out = run(cmd, input=audio_bytes, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def read_audio_from_oss(file_path: str, oss_pool: OSSPool):
    try:
        oss = oss_pool.get_conn()
        bucket = file_path[5:].split("/", 1)[0]
        name = file_path[5:].split("/", 1)[1]
        audio_bytes = oss.get_file(bucket, name).read()
        oss_pool.release_conn(oss)
        return audio_bytes
    except Exception as e:
        print(f"[SKIP] {file_path}: {e}")
        return None


def prepare_segment_16k(sample_files, audio_cache, cache_lock, oss_pool):
    metadata = json.loads(sample_files["json"])
    audio_filepath = metadata["audio_filepath"]
    start_sample = metadata["segment_start_time"]
    end_sample = metadata["segment_end_time"]
    sr = metadata["sample_rate"]

    with cache_lock:
        cached = audio_cache.get(audio_filepath)

    if cached is None:
        raw = read_audio_from_oss(audio_filepath, oss_pool)
        if raw is None:
            return None
        waveform_16k = audio_bytes_resample(raw, sr=16000)
        with cache_lock:
            audio_cache[audio_filepath] = waveform_16k
    else:
        waveform_16k = cached

    start_16k = int(start_sample * 16000 / sr)
    end_16k = int(end_sample * 16000 / sr)
    return waveform_16k[start_16k:end_16k]


# ---------------------------------------------------------------------------
# Feature extraction (all decorated with @torch.no_grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_whisper_features(model, waveform: np.ndarray) -> torch.Tensor:
    mel = log_mel_spectrogram(waveform, n_mels=model.dims.n_mels)
    mel = pad_or_trim(mel, N_FRAMES).to(model.device).to(torch.float16)
    options = whisper.DecodingOptions()
    return whisper.get_audio_features(model, mel, options)


@torch.no_grad()
def get_muq_features(model, mel_bytes: bytes, device: str) -> torch.Tensor:
    mel_npz = np.load(io.BytesIO(mel_bytes))
    mel_array = mel_npz[mel_npz.files[0]]
    mel_tensor = torch.from_numpy(mel_array).to(device)
    mel_input = {"melspec_2048": mel_tensor}
    _, hidden_states = model.model.get_predictions(mel_input, is_features_only=True)
    feat = torch.cat([hidden_states[2], hidden_states[11]], dim=-1)
    return feat.mean(dim=0, keepdim=True)[:, :750, :]


@torch.no_grad()
def get_wavlm_features(model, segment_16k: np.ndarray, device: str) -> torch.Tensor:
    wavs = torch.from_numpy(segment_16k).unsqueeze(0).to(device)
    rep, layer_results = model.extract_features(
        wavs, output_layer=model.cfg.encoder_layers, ret_layer_results=True
    )[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
    feat = torch.stack(
        [layer_reps[6], layer_reps[7], layer_reps[8], layer_reps[9]]
    ).mean(dim=0)
    if feat.shape[1] < 1500:
        feat = torch.nn.functional.pad(feat, (0, 0, 0, 1500 - feat.shape[1]))
    return feat[:, :1500, :]


# ---------------------------------------------------------------------------
# Tar reading (local path or s3:// OSS path)
# ---------------------------------------------------------------------------

def iter_tar_samples(tar_path: str):
    if tar_path.startswith("s3://"):
        oss_pool = OSSPool(max_conn=4, datatype="B")
        oss = oss_pool.get_conn()
        bucket = tar_path[5:].split("/", 1)[0]
        name = tar_path[5:].split("/", 1)[1]
        body = oss.get_file(bucket, name)
        open_kwargs = {"fileobj": body, "mode": "r|*"}
    else:
        open_kwargs = {"name": tar_path, "mode": "r|*"}

    with tarfile.open(**open_kwargs) as tar:
        current_key = None
        current_files = {}
        for m in tar:
            if not m.isfile():
                continue
            basename = m.name.split("/")[-1]
            key = basename.split(".")[0]
            suffix = basename[len(key) + 1:]
            if key != current_key:
                if current_files and current_key is not None:
                    yield current_key, current_files
                current_key = key
                current_files = {}
            f = tar.extractfile(m)
            if f:
                current_files[suffix] = f.read()
        if current_files and current_key is not None:
            yield current_key, current_files

    if tar_path.startswith("s3://"):
        oss_pool.release_conn(oss)


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

def gpu_worker(rank, sample_queue, result_queue):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    print(f"[GPU {rank}] Loading models...")
    whisper_model = whisper.load_model(WHISPER_CKPT, device=device)

    muq_model = MuQ.from_pretrained(MUQ_PRETRAINED)
    muq_model = muq_model.to(device).eval()

    checkpoint = torch.load(WAVLM_CKPT, map_location="cpu")
    cfg = WavLMConfig(checkpoint["cfg"])
    wavlm_model = WavLM(cfg)
    wavlm_model.load_state_dict(checkpoint["model"])
    wavlm_model = wavlm_model.to(device).eval()
    del checkpoint
    torch.cuda.empty_cache()
    print(f"[GPU {rank}] Models loaded.")

    oss_pool = OSSPool(max_conn=NUM_PREFETCH_THREADS + 4, datatype="B")
    audio_cache = {}
    cache_lock = threading.Lock()
    prefetch_q = queue.Queue(maxsize=PREFETCH_DEPTH)

    def prefetch_loop():
        while True:
            item = sample_queue.get()
            if item is None:
                prefetch_q.put(None)
                return
            key, files = item
            t_cpu_start = time.time()
            try:
                seg = prepare_segment_16k(files, audio_cache, cache_lock, oss_pool)
            except Exception as e:
                print(f"[GPU {rank}] [SKIP] {key}: {e}")
                seg = None
            t_cpu = time.time() - t_cpu_start
            prefetch_q.put((key, files, seg, t_cpu))

    threads = []
    for _ in range(NUM_PREFETCH_THREADS):
        t = threading.Thread(target=prefetch_loop, daemon=True)
        t.start()
        threads.append(t)

    done_threads = 0
    sample_count = 0
    while done_threads < NUM_PREFETCH_THREADS:
        item = prefetch_q.get()
        if item is None:
            done_threads += 1
            continue
        key, files, segment_16k, t_cpu = item
        if segment_16k is None:
            result_queue.put((rank, key, t_cpu, 0.0, True))
            continue

        t_gpu_start = time.time()
        get_whisper_features(whisper_model, segment_16k)
        get_muq_features(muq_model, files["mel.npz"], device)
        get_wavlm_features(wavlm_model, segment_16k, device)
        torch.cuda.synchronize(device)
        t_gpu = time.time() - t_gpu_start

        sample_count += 1
        result_queue.put((rank, key, t_cpu, t_gpu, False))
        if sample_count % 100 == 0:
            print(f"[GPU {rank}] Processed {sample_count} samples")

    for t in threads:
        t.join()
    print(f"[GPU {rank}] Done. {sample_count} samples total.")


# ---------------------------------------------------------------------------
# Main: read tar, distribute, collect timing
# ---------------------------------------------------------------------------

def main():
    global NUM_PREFETCH_THREADS
    import argparse

    parser = argparse.ArgumentParser(description="8-GPU feature extraction pipeline")
    parser.add_argument(
        "tar_path",
        nargs="?",
        default="/mnt/fcl-jfs/music_tokenizer/webdataset_data/processed_webdataset_1000w/shard_48_0_000000.tar",
        help="Path to tar (local or s3://...)",
    )
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--prefetch-threads", type=int, default=NUM_PREFETCH_THREADS)
    args = parser.parse_args()

    num_gpus = args.num_gpus
    NUM_PREFETCH_THREADS = args.prefetch_threads

    mp.set_start_method("spawn", force=True)

    sample_queues = [mp.Queue(maxsize=64) for _ in range(num_gpus)]
    result_queue = mp.Queue()

    workers = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, sample_queues[rank], result_queue),
        )
        p.start()
        workers.append(p)

    print(f"Reading tar: {args.tar_path}")
    total_distributed = 0
    t_start = time.time()
    for i, (key, files) in enumerate(iter_tar_samples(args.tar_path)):
        sample_queues[i % num_gpus].put((key, files))
        total_distributed += 1
        if total_distributed % 200 == 0:
            print(f"  Distributed {total_distributed} samples...")

    print(f"Total: {total_distributed} samples distributed.")

    for q in sample_queues:
        for _ in range(NUM_PREFETCH_THREADS):
            q.put(None)

    results = []
    skipped = 0
    for _ in range(total_distributed):
        rank, key, t_cpu, t_gpu, was_skipped = result_queue.get()
        if was_skipped:
            skipped += 1
        else:
            results.append({"key": key, "rank": rank, "t_cpu": t_cpu, "t_gpu": t_gpu})

    t_wall = time.time() - t_start

    for p in workers:
        p.join()

    if not results:
        print("No results collected.")
        return

    cpu_times = np.array([r["t_cpu"] for r in results])
    gpu_times = np.array([r["t_gpu"] for r in results])
    total_times = cpu_times + gpu_times

    def report(arr, label):
        print(f"  {label}:")
        print(f"    mean:  {arr.mean()*1000:8.1f} ms")
        print(f"    P50:   {np.percentile(arr, 50)*1000:8.1f} ms")
        print(f"    P95:   {np.percentile(arr, 95)*1000:8.1f} ms")
        print(f"    min:   {arr.min()*1000:8.1f} ms")
        print(f"    max:   {arr.max()*1000:8.1f} ms")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Benchmark: {len(results)} ok, {skipped} skipped, {num_gpus} GPUs")
    print(f"Wall clock: {t_wall:.1f} s")
    print(sep)
    report(cpu_times, "CPU preprocessing (OSS download + ffmpeg)")
    report(gpu_times, "GPU inference (whisper + muq + wavlm)")
    report(total_times, "Total per sample (CPU + GPU serial)")
    print(f"\n  Throughput:  {len(results) / t_wall:.1f} samples/sec")
    print(f"  Wall-time per sample (amortised): {t_wall / len(results) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
