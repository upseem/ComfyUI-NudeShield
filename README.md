# ComfyUI NudeShield

基于 [NudeNet](https://github.com/notAI-tech/NudeNet) 的智能 NSFW 内容检测与自动打码节点，为 ComfyUI 提供强大的内容审查功能。

## 📋 目录

- [功能特性](#功能特性)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [节点说明](#节点说明)
- [参数详解](#参数详解)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

## ✨ 功能特性

- 🔍 **智能检测**：基于 NudeNet 深度学习模型，支持 18 种敏感内容类别检测
- 🎨 **像素化打码**：高质量的马赛克处理，支持圆角和扩展区域
- ⚡ **高性能**：支持 GPU 加速（自动检测 CUDA），推理速度快
- 🎯 **精确控制**：可配置检测阈值、打码强度、区域扩展等参数
- 🔧 **灵活配置**：支持多种块数缩放策略，适应不同场景需求
- 📦 **模型支持**：支持 320n（快速）和 640m（精确）两种模型

## 📦 安装说明

### 方法一：通过 ComfyUI Manager 安装

1. 在 ComfyUI 中打开 ComfyUI Manager
2. 搜索 `ComfyUI-NudeShield` 或 `NudeShield`
3. 点击安装

### 方法二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/upseem/ComfyUI_NudeShield.git
cd ComfyUI_NudeShield
```

### 依赖安装

```bash
# 基础依赖（通常已安装）
pip install onnxruntime opencv-python numpy torch

# GPU 加速支持（可选，但推荐）
pip install onnxruntime-gpu
```

**注意**：如果使用 GPU 加速，需要确保已安装：
- NVIDIA GPU 驱动
- CUDA 工具包（与 onnxruntime-gpu 版本匹配）

### 模型下载

#### Available models

| Model | Resolution Trained | Based On | ONNX Link | PyTorch Link |
| --- | --- | --- | --- | --- |
| **320n** | 320×320 | ultralytics yolov8n | [下载 ONNX](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx) | [下载 PyTorch](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.pt) |
| **640m** | 640×640 | ultralytics yolov8m | [下载 ONNX](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx) | [下载 PyTorch](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt) |

**注意**：ComfyUI NudeShield 仅支持 ONNX 格式的模型。

#### 模型对比

| 特性 | 320n | 640m |
|------|------|------|
| **速度** | ⚡ 快速 | 🐢 较慢 |
| **精度** | ✅ 良好 | ⭐ 更高 |
| **文件大小** | 较小 | 较大 |
| **推荐场景** | 批量处理、实时检测 | 高精度需求、单张处理 |

## 🚀 快速开始

### 基本使用流程

1. **加载模型**
   - 添加 `Nudenet Model Loader` 节点
   - 选择模型文件（320n.onnx 或 640m.onnx）

2. **配置过滤标签**
   - 添加 `Filtered Labels` 节点
   - **勾选需要打码的标签**（默认都是 False，不打码）
   - 只有勾选的标签才会被打码

3. **应用检测和打码**
   - 添加 `Apply Nudenet` 节点
   - 连接模型、图像和过滤标签
   - 调整参数（详见下方参数说明）

4. **查看结果**
   - 输出图像将自动对检测到的敏感内容进行打码

## 🎛️ 节点说明

### 1. Nudenet Model Loader

**功能**：加载 NudeNet ONNX 模型

**输入**：
- `model`：模型文件名（从 `models/Nudenet/` 目录选择）

**输出**：
- `NUDENET_MODEL`：加载的模型对象

**说明**：
- 自动检测并使用 GPU（如果可用）
- 支持 320n 和 640m 两种模型
- 首次加载会显示使用的设备（GPU/CPU）

### 2. Filtered Labels

**功能**：配置需要打码的标签（黑名单）

**输入**：18 个布尔值开关，对应以下标签（**默认都是 False，不打码**）：

| 标签 | 说明 |
|------|------|
| FEMALE_GENITALIA_COVERED | 女性生殖器（已覆盖） |
| FACE_FEMALE | 女性面部 |
| BUTTOCKS_EXPOSED | 臀部（暴露） |
| FEMALE_BREAST_EXPOSED | 女性胸部（暴露） |
| FEMALE_GENITALIA_EXPOSED | 女性生殖器（暴露） |
| MALE_BREAST_EXPOSED | 男性胸部（暴露） |
| ANUS_EXPOSED | 肛门（暴露） |
| FEET_EXPOSED | 脚部（暴露） |
| BELLY_COVERED | 腹部（已覆盖） |
| FEET_COVERED | 脚部（已覆盖） |
| ARMPITS_COVERED | 腋下（已覆盖） |
| ARMPITS_EXPOSED | 腋下（暴露） |
| FACE_MALE | 男性面部 |
| BELLY_EXPOSED | 腹部（暴露） |
| MALE_GENITALIA_EXPOSED | 男性生殖器（暴露） |
| ANUS_COVERED | 肛门（已覆盖） |
| FEMALE_BREAST_COVERED | 女性胸部（已覆盖） |
| BUTTOCKS_COVERED | 臀部（已覆盖） |

**输出**：
- `FILTERED_LABELS`：标签列表（用于 Apply Nudenet 节点）

**说明**：
- ✅ **勾选（True）** = 打码（如果检测到该标签）
- ❌ **未勾选（False，默认）** = 不打码（保留原样）

### 3. Apply Nudenet

**功能**：对图像进行 NSFW 检测并自动打码

**输入**：
- `nudenet_model`：NudeNet 模型（来自 Nudenet Model Loader）
- `image`：输入图像
- `filtered_labels`：过滤标签（来自 Filtered Labels）
- 其他参数（详见下方参数详解）

**输出**：
- `IMAGE`：处理后的图像（敏感内容已打码）

## 📖 参数详解

### Apply Nudenet 节点参数

#### 1. `min_score`（最小置信度）

- **类型**：`FLOAT`
- **默认值**：`0.2`
- **范围**：`0.0 - 1.0`
- **说明**：检测结果的最低置信度阈值。只有置信度高于此值的检测结果才会被处理。
  - 值越小 → 检测越敏感（可能误检）
  - 值越大 → 检测越严格（可能漏检）
- **建议**：通常使用 `0.2`，可根据实际效果调整

#### 2. `blocks`（马赛克块数）

- **类型**：`INT`
- **默认值**：`3`
- **范围**：`1 - 100`
- **说明**：马赛克网格的块数（blocks×blocks）。**注意**：块数越多，打码效果越弱，越能看出形状。
  - **块数少**（如 3-5）→ 打码效果强，看不出形状
  - **块数多**（如 10-20）→ 打码效果弱，能看出大致轮廓
- **建议**：
  - 强打码：`3-5`
  - 中等打码：`5-10`
  - 弱打码：`10-20`

#### 3. `block_count_scaling`（块数缩放策略）

- **类型**：`STRING`（下拉选择）
- **选项**：
  - `fixed`：固定块数（所有区域使用相同的块数）
  - `fewer_when_small`：区域越小，块数越少（小区域打码更强）
  - `fewer_when_large`：区域越大，块数越少（大区域打码更强）
  - `japan`：**日本 AV 风格**（推荐）✨
    - 自动计算块数，忽略用户设置的 `blocks` 参数
    - 根据区域大小自适应，确保打码效果
    - 块数范围：5-20（保证有马赛克，且不显示明确部位）
    - 模拟日本 AV 打码风格
- **说明**：根据打码区域大小自动调整块数
- **建议**：
  - `japan`：**推荐使用**，自动适配，无需调整 `blocks` 参数
  - `fixed`：需要统一效果时使用
  - `fewer_when_small`：小区域需要更强打码时使用
  - `fewer_when_large`：大区域需要更强打码时使用

#### 4. `inference_resolution`（推理分辨率）

- **类型**：`INT`
- **默认值**：`320`
- **选项**：`320` 或 `640`
- **说明**：模型推理时使用的分辨率
  - `320`：使用 320n 模型（快速，默认）
  - `640`：使用 640m 模型（更精确但较慢）
- **建议**：
  - 批量处理：使用 `320`
  - 需要高精度：使用 `640`

#### 5. `corner_ratio`（圆角程度）

- **类型**：`FLOAT`
- **默认值**：`0.0`
- **范围**：`0.0 - 0.5`
- **说明**：马赛克块的圆角程度
  - `0.0`：矩形块（默认）
  - `0.25`：中等圆角
  - `0.5`：接近椭圆
- **建议**：通常使用 `0.0`（矩形），需要柔和效果时使用 `0.25`

#### 6. `expand_pixels`（扩展像素数）

- **类型**：`INT`
- **默认值**：`0`
- **范围**：`0 - 100`
- **说明**：在检测框四周扩展的像素数，用于扩大打码范围
  - `0`：不扩展（仅打码检测到的区域）
  - `10`：四周各扩展 10 像素
  - `20`：四周各扩展 20 像素
- **建议**：如果检测框不够准确，可以设置 `5-15` 像素的扩展

## 🔧 技术细节

### 检测流程

1. **预处理**：使用 `cv2.dnn.blobFromImage` 将图像预处理为模型输入格式
2. **推理**：在指定分辨率（320 或 640）上进行模型推理
3. **后处理**：将检测框坐标转换回原图坐标系
4. **过滤**：根据 `filtered_labels` 和 `min_score` 过滤检测结果
5. **打码**：在原始分辨率图像上对敏感区域进行像素化处理

### 性能优化

- **GPU 加速**：自动检测并使用 CUDA（如果可用）
- **高效预处理**：使用 OpenCV 的 `blobFromImage` 进行快速预处理
- **坐标转换**：精确的坐标映射，确保打码位置准确

### 与原版 NudeNet 的对比

| 特性 | 原版 NudeNet | ComfyUI NudeShield |
|------|-------------|-------------------|
| 预处理方式 | `cv2.dnn.blobFromImage` | ✅ 相同（高效） |
| 后处理方式 | `_postprocess` | ✅ 相同（精确） |
| 返回格式 | 标签名 | ✅ 相同（兼容） |
| GPU 支持 | 默认 CPU | ✅ 自动检测 GPU |
| 分辨率支持 | 固定 320 | ✅ 支持 320/640 |
| 打码方式 | 简单黑色覆盖 | ✅ 高质量像素化 |
| 集成方式 | Python 库 | ✅ ComfyUI 节点 |

## ❓ 常见问题

### Q1: 如何启用 GPU 加速？

**A**: 安装 `onnxruntime-gpu` 后，代码会自动检测并使用 GPU。首次加载模型时会显示使用的设备。

```bash
pip install onnxruntime-gpu
```

### Q2: 为什么检测不到某些内容？

**A**: 可能的原因：
1. `min_score` 设置过高，降低到 `0.1-0.2`
2. 使用 320n 模型精度不够，尝试 640m 模型
3. 图像质量过低，模型难以识别

### Q3: 打码效果不够强怎么办？

**A**: 调整以下参数：
1. 降低 `blocks` 值（如从 10 改为 3）
2. 增加 `expand_pixels`（如设置为 10-20）
3. 使用 `block_count_scaling="fixed"` 确保统一效果

### Q4: 如何只打码特定类型的内容？

**A**: 在 `Filtered Labels` 节点中：
- **勾选需要打码的标签**（默认都是 False，不打码）
- 只有勾选的标签才会被打码

例如：只打码暴露的生殖器，保留面部和脚部
- ✅ **勾选**：`FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`（需要打码）
- ❌ **不勾选**：`FACE_FEMALE`, `FACE_MALE`, `FEET_EXPOSED`, `FEET_COVERED`（保留原样）

### Q5: 320n 和 640m 模型有什么区别？

**A**:
- **320n**：快速，适合批量处理，检测精度稍低
- **640m**：精确，检测精度更高，但速度较慢

建议：先用 320n 测试，如果需要更高精度再使用 640m。

### Q6: 打码区域不准确怎么办？

**A**:
1. 增加 `expand_pixels`（如 10-20 像素）
2. 使用 640m 模型提高检测精度
3. 降低 `min_score` 以检测更多区域

### Q7: 如何模拟日本 AV 的打码风格？

**A**: **推荐使用 `japan` 模式**（最简单）：
- `block_count_scaling = "japan"` ✨ **推荐**
- 自动计算块数，无需设置 `blocks`
- 自动适配区域大小，确保打码效果

**或者手动设置**：
- `blocks = 15-25`（高密度马赛克）
- `corner_ratio = 0.0`（矩形块）
- `expand_pixels = 5-10`（稍微扩展）
- `block_count_scaling = "fixed"`（统一效果）

## 📝 更新日志

### v0.1.0
- ✅ 基于原版 NudeNet 实现
- ✅ 支持 320n 和 640m 模型
- ✅ GPU 自动加速
- ✅ 高质量像素化打码
- ✅ 圆角和区域扩展功能
- ✅ 灵活的块数缩放策略

## 📄 许可证

MIT License

## 🙏 致谢

- [NudeNet](https://github.com/notAI-tech/NudeNet) - 原始检测模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的节点式 AI 工作流框架

## 📧 联系方式

- GitHub Issues: [提交问题](https://github.com/upseem/ComfyUI_NudeShield/issues)
- Email: admin@sunx.ai

---

**注意**：本工具仅用于内容审查和合规用途，请遵守当地法律法规。

