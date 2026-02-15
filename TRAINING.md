# Codec 训练流程

基于 `load_hyperpyyaml` + torch DDP + Executor 的训练脚本，与 flowmatching2/cosyvoice 风格一致。

## 依赖

- `torch`（含 distributed）
- `hyperpyyaml`、`PyYAML`、`tqdm`、`tensorboard`
- WebDataset 时：`webdataset`

安装示例：`pip install hyperpyyaml pyyaml tqdm tensorboard webdataset`

## End-to-end 训练：损失与 EMA

### 当前使用的损失

端到端训练中，总损失由以下部分组成（在 `utils/train_utils.py` 的 `batch_forward` 中计算）：

| 损失 | 含义 | 权重（默认） | 何时使用 |
|------|------|----------------|----------|
| **flow_loss** | 条件流匹配：L1(flow_pred, flow_gt)，在 mel 有效帧上取平均 | 1.0 | 始终 |
| **commit_loss** | VQ commitment：MSE(encoder 输出, quantized.detach())，把 encoder 拉向 codebook | 0.25 | 单层 VQ 与 RVQ |
| **codebook_loss** | VQ codebook：MSE(quantized, encoder.detach())，把 codebook 拉向 encoder（SoundStream 风格） | 1.0 | 仅 RVQ |

- **单层 VQ**：`loss = flow_loss + commit_loss_weight * commit_loss`（默认 `commit_loss_weight = 0.25`）。
- **RVQ**：`loss = flow_loss + commit_loss_weight * commit_loss + codebook_loss_weight * codebook_loss`（默认 `commit_loss_weight = 0.25`、`codebook_loss_weight = 1.0`）。

flow 目标为：`flow_gt = x - (1 - 1e-5) * noise`，其中 `x` 为干净 mel，`noise` 为随机高斯；CFM 预测 `flow_pred`，与 `flow_gt` 做 L1。

### 是否使用 EMA

**当前实现中未使用 EMA（Exponential Moving Average）。**

- **Codebook EMA**：部分 codec（如 EnCodec）会对 codebook 做 EMA 更新：用 encoder 映射到每个 code 的输出的指数移动平均更新该 code 的 embedding，而不是仅靠 codebook_loss 的梯度。可提高稳定性、缓解 codebook collapse，但需维护每个 code 的统计与 decay。
- **全模型 EMA**：也有工作对整模型维护 EMA 副本用于推理（更平滑的 checkpoint），本仓库未采用。

若训练中出现码本利用率持续偏低或明显 codebook collapse，可考虑：
1. 先观察 **train/codebook_util** 与 **vq/code_usage**（TensorBoard）；
2. 适当调大 `codebook_loss_weight` 或加入 **stale code replacement**（如 mucodec descript_quantize3）；
3. 再考虑引入 **codebook EMA**（需在训练循环中按 code 聚合 encoder 输出并更新 codebook.weight）。

## 目录结构

```
codec/
├── bin/
│   ├── train.py              # 训练入口
│   ├── train_music_codec.sh   # 多卡 DDP 启动
│   └── parse_options.sh       # 命令行解析
├── conf/
│   ├── single_vq.yaml         # 单层 VQ 训练配置
│   ├── rvq_*.yaml             # RVQ 消融配置（8x1024、4096+512x7 等）
│   └── dataset_codec.yaml     # 数据集配置（WebDataset / manifest，use_mel_extractor）
├── utils/
│   ├── train_utils.py         # DDP / optimizer / scheduler / batch_forward / 码本监控
│   ├── executor.py            # 一 epoch 训练 + CV
│   └── scheduler.py           # WarmupLR, ConstantLR
├── dataset/
│   ├── codec_dataset.py       # CodecDataset、CodecWebDataset、init_dataset_and_dataloader
│   └── mel_to_features.py     # mel→波形→Whisper/WavLM/MuQ 在线特征提取（CodecFeatureExtractor）
└── model.py, flow_matching.py, ...
```

## 数据格式

### 方式一：JSONL Manifest

每行一个样本，例如：

```json
{"sample_id": "id1", "whisper_path": "/path/to/whisper.npy", "wavlm_path": "...", "muq_path": "...", "mel_path": "..."}
```

特征文件可为 `.npy`、`.pt` 或 `.npz`（mel）。形状：whisper (T50, 1280)，wavlm (T50, 1024)，muq (T25, 1024)，mel (T50, 128)。

### 方式二：WebDataset（仅 mel.npz + 在线特征）

当 tar 中只有 `{id}_seg{NN}.json` 和 `{id}_seg{NN}.mel.npz`（无预计算 whisper/wavlm/muq）时：

- 在 `dataset_codec.yaml` 中设置 `urls`（如 `.../shard_48*.tar`）、`feature_keys: { mel: "mel.npz" }`。
- 设置 **use_mel_extractor: true** 和 **feature_extractor**（含本地模型路径）：
  - `whisper_name`: 本地 Whisper 目录（如 `/mnt/yi-jfs/pretrained_models/whisper-large-v3`）
  - `wavlm_ckpt`: WavLM 目录或 .pt 路径（如 `/mnt/yi-jfs/pretrained_models/wavlm`）
  - `muq_name`: 本地 MuQ 目录（如 `/mnt/yi-jfs/pretrained_models/muq-large-msd-iter`）
  - 可选：`sample_rate`、`n_fft`、`hop_length`、`n_mels`（与 codec mel 一致）
- **特征提取位置**：
  - **DataLoader 内提取（默认）**：不设置或设置 **feature_extraction_on_gpu: false** 时，DataLoader 在取样本时只读 mel.npz，由 `CodecFeatureExtractor` 在 **CPU** 上将 mel 反演为波形，再跑 Whisper / WavLM / MuQ 得到特征，与 mel 对齐后送入模型。
  - **训练进程内 GPU 提取（推荐）**：设置 **feature_extraction_on_gpu: true** 时，DataLoader 只加载 mel（及 mel_mask），不创建特征提取器；训练进程在 **GPU** 上对每个 batch 运行 Whisper / WavLM / MuQ 后再送入模型。更快、避免 worker CPU 瓶颈与 IPC 开销。无需预先准备 whisper.npy / wavlm.npy / muq.npy。

运行训练时需将 **codec 根目录加入 PYTHONPATH**（例如在 codec 下执行 `export PYTHONPATH=$PWD`），以便 worker 能正确 `import dataset.mel_to_features` 及 whisper/wavlm/muq。

## 启动方式

在 **codec 根目录** 下执行：

```bash
export PYTHONPATH=$PWD

# 单卡
python bin/train.py --config conf/single_vq.yaml --dataset_conf conf/dataset_codec.yaml --model_dir checkpoints/exp1

# 多卡（示例 2 卡）
bash bin/train_music_codec.sh conf/single_vq.yaml exp1
# 可设置环境变量：gpus_per_node=2, batch_size=16, dataset_conf=conf/dataset_codec.yaml 等
```

**RVQ 消融**：将 `--config` 换为对应 RVQ 配置，例如：

```bash
python bin/train.py --config conf/rvq_8x1024.yaml --dataset_conf conf/dataset_codec.yaml --model_dir checkpoints/rvq_8x1024
```

## TensorBoard 与码本监控

- **Scalars**：`train/loss`、`train/flow_loss`、`train/commit_loss`、`train/codebook_util`；RVQ 时还有 `train/codebook_loss`。
- **码本利用率（防 codebook collapse）**：
  - **train/codebook_util**：当前 batch 使用的码本比例。单层 VQ 为「唯一 code 数 / codebook_size」；RVQ 为各层「唯一 code 数 / 该层 codebook_size」的平均。
  - **vq/code_usage**：code 索引使用分布直方图（每若干 step 记录一次）。

利用率持续偏低或下降可提示 codebook collapse，需关注学习率或结构。

## 损失与权重

- **单层 VQ**：`loss = flow_loss + commit_loss_weight * commit_loss`，默认 `commit_loss_weight: 0.25`。
- **RVQ**：在单层基础上增加 codebook loss（与 SoundStream / mucodec 一致）：  
  `loss = flow_loss + commit_loss_weight * commit_loss + codebook_loss_weight * codebook_loss`，  
  默认 `commit_loss_weight: 0.25`、`codebook_loss_weight: 1.0`。  
  RVQ 配置中已包含 `codebook_loss_weight: 1.0`。

## Torch DDP 与 Lightning

当前实现使用 **原生 torch DDP**（与 cosyvoice 一致），未使用 PyTorch Lightning：

- **DDP**：checkpoint 格式（model_*.pt / opt_*.pt）、resume 逻辑、训练循环完全可控。
- 若需迁移到 Lightning，可将模型封装为 LightningModule，用 Trainer 替代当前 Executor 与手写 loop。
