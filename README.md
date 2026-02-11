# 音频重建 Codec 框架

基于 Whisper / WavLM / MuQ 多级特征 + 单层 VQ 或 8 层 RVQ + 条件流匹配（CFM）的低帧率音乐编解码器。

## 架构概述

1. **语义编码**：Whisper、WavLM、MuQ 提取多级特征（可预计算或训练时从 mel 反演波形再提取）
2. **量化**：单层 VQ（256 维）或 8 层 RVQ（每层 16 维，多种 codebook 配置）
3. **解码**：条件流匹配（FlowMatchingTransformer + DiffLlama）从条件预测 mel，再经声码器得到波形

## 文件结构

```
codec/
├── model.py                 # AudioReconModel（单层 VQ / RVQ）
├── flow_matching.py         # FlowMatchingTransformer
├── llama.py                 # DiffLlama
├── config.yaml              # 模型超参
├── whisper_feature.py       # Whisper 特征（离线脚本）
├── wavlm_feature.py         # WavLM 特征（离线脚本）
├── muq_feature.py           # MuQ 特征（离线脚本）
├── dataset/
│   ├── codec_dataset.py     # CodecDataset / CodecWebDataset
│   └── mel_to_features.py   # 训练时 mel→波形→Whisper/WavLM/MuQ 在线提取
├── bin/
│   ├── train.py             # 训练入口
│   └── train_music_codec.sh # 多卡 DDP 启动
├── conf/
│   ├── single_vq.yaml      # 单层 VQ 训练配置
│   ├── dataset_codec.yaml  # 数据集（WebDataset / manifest，含 use_mel_extractor）
│   └── rvq_*.yaml         # RVQ 消融配置（8x1024、4096+512x7 等）
├── utils/
│   ├── train_utils.py       # DDP / 优化器 / batch_forward / 码本监控
│   ├── executor.py          # 一 epoch 训练 + 验证
│   └── scheduler.py         # WarmupLR, ConstantLR
├── ARCHITECTURE.md           # 架构与数据流
└── TRAINING.md              # 训练流程与数据格式
```

## 训练数据与特征

- **仅 mel.npz**：若数据只有 `mel.npz`（无预计算 whisper/wavlm/muq），在 `dataset_codec.yaml` 中设置 `use_mel_extractor: true` 和 `feature_extractor`（含本地模型路径）。DataLoader 在取样本时用 `CodecFeatureExtractor` 将 mel 反演为波形，再跑 Whisper / WavLM / MuQ 得到特征，与 mel 对齐后送入模型。
- **本地预训练模型**（示例）：
  - Whisper: `/mnt/yi-jfs/pretrained_models/whisper-large-v3`
  - WavLM: `/mnt/yi-jfs/pretrained_models/wavlm`（目录内 .pt）
  - MuQ: `/mnt/yi-jfs/pretrained_models/muq-large-msd-iter`
- **WebDataset**：tar 中每样本为 `{id}_seg{NN}.json` + `{id}_seg{NN}.mel.npz` 时，`urls` 指向 `shard_48*.tar` 等，`feature_keys: { mel: "mel.npz" }`，配合 `use_mel_extractor` 即可训练。

详见 `conf/dataset_codec.yaml` 与 `TRAINING.md`。

## 训练与监控

- **启动**（在 codec 根目录）：`export PYTHONPATH=$PWD`，然后  
  `python bin/train.py --config conf/single_vq.yaml --dataset_conf conf/dataset_codec.yaml --model_dir checkpoints/exp1`  
  多卡示例：`bash bin/train_music_codec.sh conf/single_vq.yaml exp1`。
- **TensorBoard 码本监控**（防 codebook collapse）：
  - **train/codebook_util**：当前 batch 使用的码本比例（单层 VQ 为唯一 code 数 / codebook_size；RVQ 为各层利用率取平均）。
  - **vq/code_usage**：code 索引使用分布直方图。
- **RVQ**：使用 `conf/rvq_8x1024.yaml` 等配置；RVQ 含 **codebook_loss**（与 commitment loss 一起，对齐 SoundStream/mucodec 风格），权重见 `commit_loss_weight: 0.25`、`codebook_loss_weight: 1.0`。

详见 `TRAINING.md`。

## VQ 与 RVQ 配置

- **单层 VQ**：`conf/single_vq.yaml`，`codebook_size=8192`，`codebook_dim=256`。
- **8 层 RVQ**（每层离散维度 16，时间对齐后 concat 再 `cond_proj` 进 CFM）：
  - `rvq_8x1024.yaml`：8×1024
  - `rvq_4096_512x7.yaml`：[4096, 512, 512, 512, 512, 512, 512, 512]
  - `rvq_4096_1024x7.yaml`：[4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
  - `rvq_4096_2048_1024_512x4.yaml`：[4096, 2048, 1024, 512, 512, 512, 512, 512]
  - `rvq_7layer_4096_2048_1024_512.yaml`：7 层 [4096, 2048, 1024, 512, 512, 512, 512]

RVQ 实现与 `mucodec/libs/rvq/descript_quantize3.py` 对齐：commitment + codebook loss、L2 归一化 lookup、残差逐层量化。

## 推理

- **从 codes**：`model.decode_from_codes(codes, mel_mask, n_timesteps=32, cfg=1.0, rescale_cfg=0.75)` → mel。
- **从特征**：`model.decode_from_features(whisper_feat, wavlm_feat, muq_feat, mel_mask, ...)` → mel + codes。

mel 经声码器（如 Vocos）得到波形。

## 依赖

- `torch`（含 distributed）、`hyperpyyaml`、`PyYAML`、`tqdm`、`tensorboard`
- WebDataset 训练：`webdataset`
- 使用 `use_mel_extractor` 时：Whisper / WavLM / MuQ 相关依赖，且需将 codec 根目录加入 `PYTHONPATH`

更多细节见 `ARCHITECTURE.md` 与 `TRAINING.md`。
