import io
import json
import os
import queue
import subprocess
import tarfile
import threading
import time
from subprocess import CalledProcessError, run

import numpy as np
from tqdm import tqdm
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
MONITOR_INTERVAL = 2
DEFAULT_OUTPUT_DIR = "/mnt/yi-jfs/data/codec/webdataset"


# ---------------------------------------------------------------------------
# Monitor helpers (CPU / GPU stats)
# ---------------------------------------------------------------------------

def _get_descendants(pid):
    result = [pid]
    try:
        children = subprocess.check_output(
            ["pgrep", "-P", str(pid)], text=True
        ).strip()
        for child in children.splitlines():
            result.extend(_get_descendants(int(child)))
    except subprocess.CalledProcessError:
        pass
    return result


def proc_cpu_stats(pid):
    try:
        pids = _get_descendants(pid)
        total_cpu = 0.0
        total_rss = 0
        for p in pids:
            try:
                with open(f"/proc/{p}/stat") as f:
                    fields = f.read().split()
                utime = int(fields[13])
                stime = int(fields[14])
                total_cpu += utime + stime
                with open(f"/proc/{p}/statm") as f:
                    pages = int(f.read().split()[1])
                total_rss += pages * 4096
            except (FileNotFoundError, IndexError, ProcessLookupError):
                continue
        return total_cpu, total_rss, len(pids)
    except Exception:
        return 0, 0, 0


def gpu_stats():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        idx, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
        rows.append({
            "gpu": int(idx),
            "util_pct": float(util),
            "mem_used_mb": float(mem_used),
            "mem_total_mb": float(mem_total),
        })
    return rows


# ---------------------------------------------------------------------------
# Utility: audio decoding
# ---------------------------------------------------------------------------

def audio_bytes_resample(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "1",
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


def prepare_segment_16k(sample_files, song_cache, cache_lock, oss_pool):
    metadata = json.loads(sample_files["json"])
    audio_filepath = metadata["audio_filepath"]
    start_sample = metadata["segment_start_time"]
    end_sample = metadata["segment_end_time"]
    sr = metadata["sample_rate"]

    with cache_lock:
        waveform_16k = song_cache.get(audio_filepath)

    if waveform_16k is None:
        raw = read_audio_from_oss(audio_filepath, oss_pool)
        if raw is None:
            return None
        waveform_16k = audio_bytes_resample(raw, sr=16000)
        with cache_lock:
            song_cache.clear()
            song_cache[audio_filepath] = waveform_16k

    start_16k = int(start_sample * 16000 / sr)
    end_16k = int(end_sample * 16000 / sr)
    return waveform_16k[start_16k:end_16k]


# ---------------------------------------------------------------------------
# Feature extraction
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
# Tar I/O helpers
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


def load_tar_list(webdataset_path: str, input_dir: str, max_tars: int):
    """Load first max_tars tar paths from webdataset manifest."""
    with open(webdataset_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    paths = []
    for line in lines[:max_tars]:
        if line.startswith("s3://") or line.startswith("/"):
            paths.append(line)
        else:
            paths.append(os.path.join(input_dir, line))
    return paths


def _add_bytes_to_tar(tar_out, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = time.time()
    tar_out.addfile(info, io.BytesIO(data))


def _numpy_to_npz_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, arr)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GPU worker (extracts features and writes output tar shard)
# ---------------------------------------------------------------------------

def gpu_worker(rank, sample_queue, result_queue, output_dir):
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
    song_cache = {}
    cache_lock = threading.Lock()
    prefetch_q = queue.Queue(maxsize=PREFETCH_DEPTH)

    def prefetch_loop():
        while True:
            item = sample_queue.get()
            if item is None:
                prefetch_q.put(None)
                return
            key, files = item
            try:
                seg = prepare_segment_16k(files, song_cache, cache_lock, oss_pool)
            except Exception as e:
                print(f"[GPU {rank}] [SKIP] {key}: {e}")
                seg = None
            prefetch_q.put((key, files, seg))

    threads = []
    for _ in range(NUM_PREFETCH_THREADS):
        t = threading.Thread(target=prefetch_loop, daemon=True)
        t.start()
        threads.append(t)

    output_tar_path = os.path.join(output_dir, f"shard_{rank:03d}.tar")
    done_threads = 0
    sample_count = 0

    with tarfile.open(output_tar_path, "w") as out_tar:
        while done_threads < NUM_PREFETCH_THREADS:
            item = prefetch_q.get()
            if item is None:
                done_threads += 1
                continue
            key, files, segment_16k = item
            if segment_16k is None:
                result_queue.put((rank, key, True))
                continue

            whisper_feat = get_whisper_features(whisper_model, segment_16k)
            muq_feat = get_muq_features(muq_model, files["mel.npz"], device)
            wavlm_feat = get_wavlm_features(wavlm_model, segment_16k, device)
            torch.cuda.synchronize(device)

            for suffix, data in files.items():
                _add_bytes_to_tar(out_tar, f"{key}.{suffix}", data)
            _add_bytes_to_tar(
                out_tar, f"{key}.whisper.npz",
                _numpy_to_npz_bytes(whisper_feat.cpu().half().numpy()),
            )
            _add_bytes_to_tar(
                out_tar, f"{key}.muq.npz",
                _numpy_to_npz_bytes(muq_feat.cpu().half().numpy()),
            )
            _add_bytes_to_tar(
                out_tar, f"{key}.wavlm.npz",
                _numpy_to_npz_bytes(wavlm_feat.cpu().half().numpy()),
            )

            sample_count += 1
            result_queue.put((rank, key, False))
            if sample_count % 1000 == 0:
                print(f"[GPU {rank}] Processed {sample_count} samples", flush=True)

    for t in threads:
        t.join()
    print(f"[GPU {rank}] Done. {sample_count} samples → {output_tar_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global NUM_PREFETCH_THREADS
    import argparse

    parser = argparse.ArgumentParser(description="8-GPU feature extraction pipeline")
    parser.add_argument(
        "tar_path",
        nargs="?",
        default=None,
        help="Path to single tar (local or s3://...). Ignored if --webdataset is set.",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--webdataset", type=str,
                        default=os.path.join(script_dir, "webdataset"),
                        help="Path to manifest file listing tars (one per line). Uses first 1000.")
    parser.add_argument("--input-dir", type=str,
                        default="/mnt/fcl-jfs/music_tokenizer/webdataset_data/processed_webdataset_1000w",
                        help="Base dir for tar paths when lines in --webdataset are relative.")
    parser.add_argument("--max-tars", type=int, default=1000,
                        help="Max number of tars to process from --webdataset (default: 1000).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write output tar shards")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--prefetch-threads", type=int, default=NUM_PREFETCH_THREADS)
    args = parser.parse_args()

    num_gpus = args.num_gpus
    NUM_PREFETCH_THREADS = args.prefetch_threads
    os.makedirs(args.output_dir, exist_ok=True)

    if args.webdataset:
        tar_paths = load_tar_list(args.webdataset, args.input_dir, args.max_tars)
        if not tar_paths:
            print("No tar paths loaded from webdataset.")
            return
        print(f"Processing {len(tar_paths)} tars from {args.webdataset}")
    else:
        single = args.tar_path or "/mnt/fcl-jfs/music_tokenizer/webdataset_data/processed_webdataset_1000w/shard_48_0_000000.tar"
        tar_paths = [single]
        print(f"Processing single tar: {single}")

    print(f"Output dir: {args.output_dir}")

    mp.set_start_method("spawn", force=True)

    sample_queues = [mp.Queue(maxsize=64) for _ in range(num_gpus)]
    result_queue = mp.Queue()

    workers = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, sample_queues[rank], result_queue, args.output_dir),
        )
        p.start()
        workers.append(p)

    # Distribution phase
    total_distributed = 0
    t_start = time.time()
    current_song = None
    current_gpu = 0

    with tqdm(tar_paths, desc="Reading tars", unit="tar") as pbar_tars:
        for tar_path in pbar_tars:
            pbar_tars.set_postfix_str(f"samples={total_distributed}")
            for key, files in iter_tar_samples(tar_path):
                metadata = json.loads(files["json"])
                song = metadata["audio_filepath"]
                if song != current_song:
                    current_song = song
                    current_gpu = (current_gpu + 1) % num_gpus
                sample_queues[current_gpu].put((key, files))
                total_distributed += 1

    for q in sample_queues:
        for _ in range(NUM_PREFETCH_THREADS):
            q.put(None)

    # Collection phase with tqdm and CPU/GPU monitor
    pid = os.getpid()
    hz = os.sysconf("SC_CLK_TCK")
    prev_cpu_ticks = 0
    prev_time = time.time()
    monitor_stop = threading.Event()
    stats_str = [""]

    def monitor_loop():
        nonlocal prev_cpu_ticks, prev_time
        while not monitor_stop.is_set():
            monitor_stop.wait(MONITOR_INTERVAL)
            if monitor_stop.is_set():
                break
            now = time.time()
            dt = now - prev_time
            cpu_ticks, rss_bytes, num_procs = proc_cpu_stats(pid)
            delta_ticks = cpu_ticks - prev_cpu_ticks
            cpu_pct = (delta_ticks / hz) / dt * 100 if dt > 0 else 0
            prev_cpu_ticks = cpu_ticks
            prev_time = now
            rss_gb = rss_bytes / (1024 ** 3)
            gpus = gpu_stats()
            gpu_utils = [g["util_pct"] for g in sorted(gpus, key=lambda x: x["gpu"])]
            gpu_mem_gb = sum(g["mem_used_mb"] for g in gpus) / 1024
            gpu_strs = " ".join(f"GPU{i}:{u:3.0f}%" for i, u in enumerate(gpu_utils[:8]))
            stats_str[0] = (
                f"procs={num_procs} CPU={cpu_pct:.0f}% RSS={rss_gb:.1f}GB "
                f"{gpu_strs} VRAM={gpu_mem_gb:.1f}GB"
            )

    mon_thread = threading.Thread(target=monitor_loop, daemon=True)
    mon_thread.start()

    processed = 0
    skipped = 0
    with tqdm(total=total_distributed, desc="Processing", unit="sample", smoothing=0.1) as pbar:
        for _ in range(total_distributed):
            rank, key, was_skipped = result_queue.get()
            if was_skipped:
                skipped += 1
            else:
                processed += 1
            pbar.update(1)
            if stats_str[0]:
                pbar.set_postfix_str(stats_str[0], refresh=False)

    monitor_stop.set()
    mon_thread.join(timeout=MONITOR_INTERVAL + 1)

    t_wall = time.time() - t_start

    for p in workers:
        p.join()

    print(f"\n{'=' * 60}")
    print(f"Done: {processed} saved, {skipped} skipped, {num_gpus} GPUs")
    print(f"Wall clock: {t_wall:.1f}s  ({processed / t_wall:.1f} samples/sec)")
    print(f"Output: {args.output_dir}/shard_{{000..{num_gpus-1:03d}}}.tar")


if __name__ == "__main__":
    main()
