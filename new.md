# NudeNet 原版 vs ComfyUI 实现对比分析

## 源码参考
- 原版源码：https://github.com/notAI-tech/NudeNet/blob/v3/nudenet/nudenet.py
- 默认模型：`320n.onnx`（位于包目录下）

---

## 核心区别总结

### 1. 图像预处理方式

#### 原版 (`_read_image`)
```python
# 使用 cv2.dnn.blobFromImage 进行预处理
input_blob = cv2.dnn.blobFromImage(
    mat_pad,
    1 / 255.0,  # normalize
    (target_size, target_size),  # resize to model input size
    (0, 0, 0),  # mean subtraction
    swapRB=True,  # swap red and blue channels
    crop=False,  # don't crop
)
```

**特点**：
- 使用 OpenCV 的 `blobFromImage` 函数，自动处理归一化和通道转换
- 支持多种输入类型：`str`（文件路径）、`np.ndarray`、`bytes`、`_io.BufferedReader`
- Padding 策略：先 pad 到正方形（max_size），再 resize 到 target_size
- 返回格式：`(input_blob, x_ratio, y_ratio, x_pad, y_pad, image_original_width, image_original_height)`

#### 您的实现 (`read_image`)
```python
# 手动进行 resize 和 padding
img = cv2.resize(img, (new_width, new_height))
img = cv2.copyMakeBorder(...)
img = cv2.resize(img, (target_size, target_size))
image_data = np.transpose(img, (2, 0, 1))
```

**特点**：
- 只支持 `np.ndarray` 输入
- 手动进行 resize、padding 和转置操作
- Padding 策略：先 resize 保持宽高比，再 pad 到 target_size，最后 resize 到 target_size
- 返回格式：`(image_data, resize_factor, pad_left, pad_top)`

**区别**：
- 原版使用 `blobFromImage` 更高效，自动处理归一化
- 您的版本手动处理，更灵活但代码更复杂
- Padding 计算方式不同

---

### 2. 坐标转换方式

#### 原版 (`_postprocess`)
```python
# 使用 x_ratio, y_ratio, x_pad, y_pad 进行坐标转换
x = x * (image_original_width + x_pad) / model_width
y = y * (image_original_height + y_pad) / model_height
w = w * (image_original_width + x_pad) / model_width
h = h * (image_original_height + y_pad) / model_height
```

**特点**：
- 使用 `x_ratio` 和 `y_ratio`（基于 max_size 计算）
- 坐标转换时考虑 padding 的影响
- 硬编码 `min_score=0.2`

#### 您的实现 (`postprocess`)
```python
# 使用 resize_factor, pad_left, pad_top 进行坐标转换
left = int(round((x - w * 0.5 - pad_left) * resize_factor))
top = int(round((y - h * 0.5 - pad_top) * resize_factor))
width = int(round(w * resize_factor))
height = int(round(h * resize_factor))
```

**特点**：
- 使用 `resize_factor`（基于对角线长度计算）
- 坐标转换时先减去 padding，再乘以 resize_factor
- 可配置 `min_score` 参数

**区别**：
- 原版使用比例缩放，您的版本使用 resize_factor
- 坐标转换公式不同，但目标相同（将模型输出坐标转换回原图坐标）

---

### 3. 输出格式

#### 原版
```python
detections.append({
    "class": __labels[class_id],  # 返回标签名称
    "score": float(score),
    "box": [int(x), int(y), int(w), int(h)],
})
```

#### 您的实现
```python
{
    "id": class_ids[i],  # 返回类别 ID
    "score": round(float(scores[i]), 2),
    "box": boxes[i],
}
```

**区别**：
- 原版返回标签名称（如 "FACE_FEMALE"）
- 您的版本返回类别 ID（如 1）
- 您的版本更适合 ComfyUI 的标签过滤机制

---

### 4. 模型加载方式

#### 原版
```python
class NudeDetector:
    def __init__(self, model_path=None, providers=None, inference_resolution=320):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "320n.onnx")
            if not model_path
            else model_path,
        )
```

**特点**：
- 默认模型：`320n.onnx`（位于包目录下）
- 固定推理分辨率：320
- 提供 `detect`、`detect_batch`、`censor` 方法

#### 您的实现
```python
class NudenetModelLoader:
    def load_model(self, model, providers="CPU"):
        self.session = onnxruntime.InferenceSession(
            os.path.join(current_paths[0], model),  # 从 ComfyUI 模型目录加载
            providers=C.get_available_providers()
        )
        # 动态获取模型输入尺寸
        self.input_width = model_inputs[0].shape[2]
        self.input_height = model_inputs[0].shape[3]
```

**特点**：
- 从 ComfyUI 的模型目录加载（`models/Nudenet/`）
- 动态获取模型输入尺寸（支持不同分辨率的模型）
- 集成到 ComfyUI 节点系统

**区别**：
- 原版：固定模型路径和分辨率
- 您的版本：灵活支持不同模型和分辨率

---

### 5. 功能扩展

#### 原版功能
- `detect(image_path)`: 单张图像检测
- `detect_batch(image_paths, batch_size=4)`: 批量检测
- `censor(image_path, classes=[], output_path=None)`: 打码功能（纯黑色覆盖）

#### 您的实现功能
- `ApplyNudenet` 节点：ComfyUI 节点接口
- `FilteredLabel` 节点：标签过滤配置
- `NudenetModelLoader` 节点：模型加载
- **像素化处理**：集成 `pixelate` 函数，支持可配置的像素块数量
- **块数缩放策略**：`fixed`、`fewer_when_small`、`fewer_when_large`

**区别**：
- 原版：独立的 Python 库，提供基础检测和简单打码
- 您的版本：ComfyUI 集成，提供像素化打码和灵活的配置选项

---

### 6. 代码结构

#### 原版结构
```
nudenet.py
├── __labels (私有标签列表)
├── _read_image() (私有预处理函数)
├── _postprocess() (私有后处理函数)
└── NudeDetector (主类)
    ├── __init__()
    ├── detect()
    ├── detect_batch()
    └── censor()
```

#### 您的实现结构
```
nodes.py
├── CLASSIDS_LABELS_MAPPING (公开标签映射)
├── LABELS_CLASSIDS_MAPPING (反向映射)
├── BLOCK_COUNT_BEHAVIOURS (块数缩放策略)
├── read_image() (公开预处理函数)
├── postprocess() (公开后处理函数)
├── nudenet_execute() (执行函数，集成像素化)
└── ComfyUI 节点类
    ├── ApplyNudenet
    ├── FilteredLabel
    └── NudenetModelLoader
```

**区别**：
- 原版：面向库的设计，函数私有化
- 您的版本：面向 ComfyUI 节点，函数公开化

---

## 关键算法差异

### 1. Padding 计算

**原版**：
```python
max_size = max(mat_c3.shape[:2])  # 取宽高的最大值
x_pad = max_size - mat_c3.shape[1]
y_pad = max_size - mat_c3.shape[0]
# 先 pad 到正方形，再 resize
```

**您的版本**：
```python
# 先 resize 保持宽高比
if img_height > img_width:
    new_height = target_size
    new_width = int(round(target_size * aspect))
# 再 pad 到 target_size
pad_x = target_size - new_width
pad_y = target_size - new_height
# 最后 resize 到 target_size
```

**影响**：
- 原版：可能产生更大的中间图像（max_size ≥ target_size）
- 您的版本：中间图像更接近 target_size，可能更节省内存

### 2. 坐标转换公式

**原版**：
```python
# 模型输出是中心坐标
x = x - w / 2  # 转换为左上角
y = y - h / 2
# 缩放回原图尺寸
x = x * (image_original_width + x_pad) / model_width
```

**您的版本**：
```python
# 模型输出是中心坐标
left = int(round((x - w * 0.5 - pad_left) * resize_factor))
top = int(round((y - h * 0.5 - pad_top) * resize_factor))
```

**影响**：
- 两种方法在数学上等价，但实现细节不同
- 您的版本使用 `resize_factor`（基于对角线），可能在某些情况下更精确

---

## 性能对比

| 方面 | 原版 | 您的实现 |
|------|------|----------|
| **预处理速度** | 更快（使用 blobFromImage） | 稍慢（手动处理） |
| **内存占用** | 可能更高（max_size padding） | 可能更低（target_size padding） |
| **灵活性** | 较低（固定分辨率） | 较高（支持不同分辨率模型） |
| **集成度** | 独立库 | ComfyUI 深度集成 |

---

## 建议

1. **预处理优化**：可以考虑使用 `cv2.dnn.blobFromImage` 来提升预处理速度
2. **保持兼容性**：当前的实现已经很好地适配了 ComfyUI 的工作流
3. **模型支持**：您的实现支持不同分辨率的模型，这是一个优势

---

## 总结

您的实现相比原版：
- ✅ **更灵活**：支持不同分辨率的模型，可配置参数更多
- ✅ **更集成**：深度集成到 ComfyUI 节点系统
- ✅ **功能更丰富**：提供像素化打码和块数缩放策略
- ⚠️ **预处理稍慢**：手动处理比 `blobFromImage` 稍慢
- ✅ **更适合 ComfyUI**：输出格式（ID 而非标签名）更适合节点系统

原版更适合作为独立的 Python 库使用，而您的实现更适合 ComfyUI 工作流场景。

