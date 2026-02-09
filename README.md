# HeartCodec实现

基于HeartMuLa论文实现的低帧率音乐编解码器框架（单层VQ版本）

## 架构概述

HeartCodec包含三个主要部分：

1. **语义丰富编码器**：使用Whisper、WavLM和MuQ提取多级特征
2. **超低帧率压缩器**：特征融合、查询下采样和单层VQ量化
3. **高保真重建解码器**：条件流匹配模型 + EAR_VAE解码器

## 文件结构

```
codec/
├── whisper_feature.py      # Whisper特征提取
├── muq_feature.py          # MuQ特征提取
├── wavlm_feature.py        # WavLM特征提取
├── parallel.sh             # 并行特征提取脚本
├── heartcodec.py           # HeartCodec编码器（单层VQ）
├── flow_matching_decoder.py # 条件流匹配解码器
└── example_usage.py        # 使用示例
```

## 快速开始

### 1. 并行提取特征

```bash
# 处理单个文件
./parallel.sh /path/to/audio.wav /path/to/output

# 处理整个目录
./parallel.sh /path/to/audio_dir /path/to/output

# 指定模型路径
./parallel.sh /path/to/audio.wav /path/to/output \
    /mnt/yi-jfs/pretrained_models/whisper-large-v3 \
    /mnt/yi-jfs/pretrained_models/muq-large-msd-iter \
    /path/to/wavlm.pt
```

### 2. 使用HeartCodec进行编码/解码

```python
from heartcodec import HeartCodec
from flow_matching_decoder import ConditionalFlowMatching, HeartCodecDecoder
import torch

# 1. 定义特征维度和帧率
feature_dims = {
    'y1': 1024,  # MuEncoder语义特征
    'y2': 768,   # WavLM特征
    'y3': 1280,  # Whisper特征
    'y4': 1024,  # MuEncoder声学特征
}

feature_frame_rates = {
    'y1': 25.0,
    'y2': 50.0,
    'y3': 50.0,
    'y4': 25.0,
}

# 2. 创建编码器
heartcodec = HeartCodec(
    feature_dims=feature_dims,
    feature_frame_rates=feature_frame_rates,
    fusion_dim=512,
    codebook_size=8192,
    target_frame_rate=25.0,
    low_frame_rate=12.5
)

# 3. 创建流匹配模型
flow_matching = ConditionalFlowMatching(
    latent_dim=64,
    condition_dim=512,
    hidden_dim=512,
    num_layers=8,
    num_heads=8
)

# 4. 创建完整解码器
decoder = HeartCodecDecoder(
    heartcodec=heartcodec,
    flow_matching=flow_matching,
    ear_vae_path=None  # 自动从HuggingFace下载
)

# 5. 编码
features = {
    'y1': y1_tensor,  # 从MuQ提取
    'y2': y2_tensor,  # 从WavLM提取
    'y3': y3_tensor,  # 从Whisper提取
    'y4': y4_tensor,  # 从MuQ提取
}
y_l_quantized, indices, commit_loss = heartcodec(features, feature_frame_rates)

# 6. 解码
reconstructed_audio = decoder.decode_from_tokens(indices, num_steps=50)
```

## 主要组件说明

### SingleLayerVQ
单层向量量化模块，将连续特征量化为离散token。

### FeatureFusion
特征融合模块，将多个不同帧率的特征序列融合为统一的25Hz表示。

### QueryDownsampler
查询下采样模块，使用可学习的query token将帧率从25Hz降到12.5Hz。

### ConditionalFlowMatching
条件流匹配模型，从量化特征重建VAE潜在空间。

### HeartCodecDecoder
完整解码器，整合编码器、流匹配和EAR_VAE，实现端到端的音频重建。

## 注意事项

1. **模型路径**：确保提供正确的预训练模型路径
   - Whisper: `/mnt/yi-jfs/pretrained_models/whisper-large-v3`
   - MuQ: `/mnt/yi-jfs/pretrained_models/muq-large-msd-iter`
   - WavLM: 需要提供checkpoint路径

2. **特征维度**：根据实际使用的模型调整`feature_dims`和`feature_frame_rates`

3. **EAR_VAE**：解码器会自动尝试从HuggingFace下载EAR_VAE模型，也可以手动指定路径

4. **设备**：默认使用CUDA，如果没有GPU会自动切换到CPU

## 与论文的差异

本实现使用**单层VQ**替代论文中的**多层RVQ**，简化了tokenization架构，便于快速原型和实验。

## 训练

训练代码需要实现：
- 特征对齐损失（feature alignment loss）
- 流匹配训练循环
- EAR_VAE微调

参考`flow_matching_decoder.py`中的`compute_loss`方法。

## 许可证

请参考HeartMuLa论文和相关代码库的许可证。
