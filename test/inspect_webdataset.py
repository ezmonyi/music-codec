#!/usr/bin/env python3
"""Inspect webdataset tar shard structure: list samples and keys per sample."""

import argparse
import glob
import json
import tarfile


def main():
    parser = argparse.ArgumentParser(description="Inspect webdataset tar structure")
    parser.add_argument(
        "path",
        nargs="?",
        default="/mnt/fcl-jfs/music_tokenizer/webdataset_data/processed_webdataset_1000w/shard_48*.tar",
        help="Glob path to tar shards (e.g. .../shard_48*.tar)",
    )
    parser.add_argument("-n", "--max-samples", type=int, default=3, help="Max samples to show detail (default 3)")
    parser.add_argument("--max-shards", type=int, default=2, help="Max shards to open (default 2)")
    args = parser.parse_args()

    shards = sorted(glob.glob(args.path))
    if not shards:
        print(f"No shards found for: {args.path}")
        return
    print(f"Found {len(shards)} shard(s). Opening up to {args.max_shards}.\n")

    for shard_path in shards[: args.max_shards]:
        print(f"=== {shard_path} ===")
        try:
            with tarfile.open(shard_path, "r|*") as tar:
                # Group members by sample id: basename before first dot (e.g. id_seg00)
                current_key = None
                current_files = []
                sample_count = 0
                for m in tar:
                    if not m.isfile():
                        continue
                    name = m.name
                    basename = name.split("/")[-1]
                    key = basename.split(".")[0] if "." in basename else basename
                    ext = basename.rsplit(".", 1)[-1].lower() if "." in basename else ""
                    if key != current_key:
                        if current_files:
                            if sample_count < args.max_samples:
                                _print_sample(current_files, sample_count)
                            sample_count += 1
                        current_key = key
                        current_files = []
                    json_content = None
                    if ext == "json":
                        try:
                            f = tar.extractfile(m)
                            if f:
                                json_content = f.read().decode("utf-8")
                        except Exception:
                            pass
                    current_files.append((name, m.size, ext, json_content))

                if current_files:
                    if sample_count < args.max_samples:
                        _print_sample(current_files, sample_count)
                    sample_count += 1

                print(f"[Shard total: {sample_count} samples]\n")
        except Exception as e:
            print(f"Error: {e}\n")

    print("Done.")


def _print_sample(files, sample_count):
    """Print one sample's file list (name, size, ext) and .json content."""
    print(f"  Sample {sample_count}:")
    for item in sorted(files, key=lambda x: x[0]):
        name, size, ext = item[0], item[1], item[2]
        json_content = item[3] if len(item) > 3 else None
        print(f"    {name}  ({size} bytes, .{ext})")
        if json_content is not None:
            try:
                obj = json.loads(json_content)
                print(f"      --- json ---")
                print(json.dumps(obj, indent=4, ensure_ascii=False))
            except Exception:
                print(f"      --- raw ---\n{json_content}")
    print()


if __name__ == "__main__":
    main()
