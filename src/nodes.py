import os
import math
import cv2
import torch
import numpy as np
from typing_extensions import TypedDict
from onnxruntime import InferenceSession

import folder_paths
from ..utils import pixelate

if "Nudenet" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "Nudenet")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["Nudenet"]
folder_paths.supported_pt_extensions.update({".onnx"})
folder_paths.folder_names_and_paths["Nudenet"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)

class ModelLoader(TypedDict):
    session: InferenceSession
    input_width: int
    input_height: int
    input_name: str

CLASSIDS_LABELS_MAPPING = {
    0: "FEMALE_GENITALIA_COVERED",
    1: "FACE_FEMALE",
    2: "BUTTOCKS_EXPOSED",
    3: "FEMALE_BREAST_EXPOSED",
    4: "FEMALE_GENITALIA_EXPOSED",
    5: "MALE_BREAST_EXPOSED",
    6: "ANUS_EXPOSED",
    7: "FEET_EXPOSED",
    8: "BELLY_COVERED",
    9: "FEET_COVERED",
    10: "ARMPITS_COVERED",
    11: "ARMPITS_EXPOSED",
    12: "FACE_MALE",
    13: "BELLY_EXPOSED",
    14: "MALE_GENITALIA_EXPOSED",
    15: "ANUS_COVERED",
    16: "FEMALE_BREAST_COVERED",
    17: "BUTTOCKS_COVERED",
}

LABELS_CLASSIDS_MAPPING = labels_classids_mapping = {
    label: class_id for class_id, label in CLASSIDS_LABELS_MAPPING.items()
}

BLOCK_COUNT_BEHAVIOURS = [
    "fixed",
    "fewer_when_small",
    "fewer_when_large",
    "japan"  # 日本 AV 风格：自适应块数，保证打码效果，忽略用户设置的 blocks
]

def read_image(img, target_size=320):
    """
    使用原版 NudeNet 的高效预处理方式（cv2.dnn.blobFromImage）
    适配 ComfyUI 的输入格式（numpy array, [H, W, C], 0-1 float32）
    """
    assert isinstance(img, np.ndarray)

    # ComfyUI 输入是 [H, W, C]，RGB，0-1 float32
    # 转换为 BGR，uint8 格式（原版期望的格式）
    image_original_height, image_original_width = img.shape[:2]

    # 转换为 uint8 (0-255)
    mat_c3 = (img * 255.0).astype(np.uint8)
    # RGB -> BGR
    mat_c3 = cv2.cvtColor(mat_c3, cv2.COLOR_RGB2BGR)

    # 原版策略：先 pad 到正方形（max_size），再 resize 到 target_size
    max_size = max(mat_c3.shape[:2])  # get max size from width and height
    x_pad = max_size - mat_c3.shape[1]  # set xPadding
    x_ratio = max_size / mat_c3.shape[1]  # set xRatio
    y_pad = max_size - mat_c3.shape[0]  # set yPadding
    y_ratio = max_size / mat_c3.shape[0]  # set yRatio

    # Pad to square
    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    # 使用 cv2.dnn.blobFromImage 进行高效预处理
    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,  # normalize (虽然已经是 uint8，但 blobFromImage 需要归一化)
        (target_size, target_size),  # resize to model input size
        (0, 0, 0),  # mean subtraction
        swapRB=True,  # swap red and blue channels (BGR -> RGB)
        crop=False,  # don't crop
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def postprocess(
    output,
    x_pad,
    y_pad,
    x_ratio,
    y_ratio,
    image_original_width,
    image_original_height,
    model_width,
    model_height,
    min_score,
):
    """
    使用原版 NudeNet 的后处理方式（_postprocess）
    保留可配置的 min_score 参数（原版硬编码为 0.2）
    返回格式与原版一致：使用 class 标签名（而非 id）
    这样如果原版更新了标签，只需更新 CLASSIDS_LABELS_MAPPING 即可
    """
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= min_score:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0:4]

            # Convert from center coordinates to top-left corner coordinates
            x = x - w / 2
            y = y - h / 2

            # Scale coordinates to original image size
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height

            # Clip coordinates to image boundaries
            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])  # 保持 float，最后再转 int（与原版一致，更精确）

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        x, y, w, h = box
        detections.append(
            {
                "class": CLASSIDS_LABELS_MAPPING[class_id],  # 使用标签名（与原版一致）
                "score": round(float(score), 2),
                "box": [int(x), int(y), int(w), int(h)],  # 最后转 int（与原版一致）
            }
        )

    return detections


def nudenet_execute(
    nudenet_model: ModelLoader,
    input_images: torch.Tensor,
    filtered_labels: list,
    min_score: float,
    blocks: float,
    block_count_scaling: str,
    inference_resolution: int = None,
    corner_ratio: float = 0.0,
    expand_pixels: int = 0,
):
    """
    inference_resolution: 推理分辨率（320 或 640）
    如果为 None，则使用模型的默认分辨率（input_width）
    原版支持 320n 和 640m 两种模型，640m 检测精度更高但速度较慢

    corner_ratio: 圆角程度（0.0-0.5），0=矩形，越大越接近椭圆
    expand_pixels: 扩展马赛克区域的像素数，用于扩大打码范围
    """
    input_images = input_images.clone()

    assert isinstance(
        input_images, torch.Tensor), "input_images must be a torch.Tensor but got %s" % type(input_images)
    assert input_images.dim(
    ) == 4, "input_images must be a 4D tensor but got %s" % input_images.dim()

    # 确定推理分辨率
    if inference_resolution is None:
        inference_resolution = nudenet_model["input_width"]
    else:
        # 如果指定了推理分辨率，使用指定的值（允许覆盖模型默认值）
        model_default = nudenet_model["input_width"]
        if inference_resolution != model_default:
            print(f"[nudenet_execute] Using inference_resolution={inference_resolution} (model default: {model_default})")
            print(f"[nudenet_execute] Note: 320n model recommended for 320, 640m model recommended for 640")
        else:
            print(f"[nudenet_execute] Using inference_resolution={inference_resolution} (matches model default)")

    batch_size = input_images.shape[0]
    output_images = []

    for i in range(batch_size):
        # 逐张处理批次中的图像。ComfyUI 的 IMAGE 通常为 [H, W, C]，float32。
        image = input_images[i].cpu().numpy()
        # 预处理：使用原版高效方式（cv2.dnn.blobFromImage）
        preprocessed_image, x_ratio, y_ratio, x_pad, y_pad, image_original_width, image_original_height = read_image(
            image, inference_resolution  # 使用指定的推理分辨率
        )
        # 推理：使用 ONNX Runtime 运行模型，输入名来自载入阶段解析的 input_name。
        outputs = nudenet_model["session"].run(
            None, {nudenet_model["input_name"]: preprocessed_image}
        )
        # 后处理：将模型输出的框变换回原图坐标系，并按 min_score 过滤 + NMS。
        # 使用推理分辨率作为模型尺寸（因为预处理时使用了 inference_resolution）
        detections = postprocess(
            outputs,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            inference_resolution,  # 使用推理分辨率
            inference_resolution,  # 使用推理分辨率
            min_score,
        )
        # 仅处理在 filtered_labels 列表中的检测目标（勾选的标签才会被打码）
        # filtered_labels 包含用户勾选的需要打码的标签
        censored = [d for d in detections if d.get(
            "class") in filtered_labels]

        # 初始块数，以 blocks 为基准；若为 fixed 策略则后续不再缩放。
        if block_count_scaling == "fixed":
            scaled_blocks = blocks

        # 遍历需要被处理的目标区域，对每个框单独执行像素化处理。
        image_height, image_width = image.shape[:2]

        for d in censored:
            box = d["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # 扩展区域（expand_pixels）
            x_expanded = max(0, x - expand_pixels)
            y_expanded = max(0, y - expand_pixels)
            w_expanded = min(image_width - x_expanded, w + expand_pixels * 2)
            h_expanded = min(image_height - y_expanded, h + expand_pixels * 2)

            # 确保扩展后的区域有效
            if w_expanded > 0 and h_expanded > 0:
                area = image[y_expanded: y_expanded + h_expanded, x_expanded: x_expanded + w_expanded]

                # 非固定策略下，根据目标框占图像的比例对 blocks 做自适应缩放。
                # d_pct 表示在高或宽方向上的相对占比，用于估算区域大小。
                if block_count_scaling == "japan":
                    # 日本 AV 风格：自适应块数，忽略用户设置的 blocks
                    # 使用平滑的自适应算法，根据区域大小动态计算块数和块大小
                    # 目标：块更大，块数更少，确保打码效果
                    avg_size = (w_expanded + h_expanded) / 2  # 区域平均尺寸

                    # 自适应目标块大小：使用对数函数，区域越大块越大，但增长逐渐放缓
                    # 范围：20-50 像素（增大范围，让块更大）
                    # 公式：target_block_size = 20 + 30 * (1 - exp(-avg_size/500))
                    # 这样：小区域(50px)≈20px，中区域(200px)≈28px，大区域(1000px)≈45px
                    target_block_size = 20 + 30 * (1 - math.exp(-avg_size / 500))
                    target_block_size = max(20, min(50, target_block_size))  # 限制在 20-50 之间

                    calculated_blocks = int(avg_size / target_block_size)

                    # 自适应最大块数：使用对数函数，区域越大允许更多块数
                    # 范围：5-25 块（减少最大块数，让块更大）
                    # 公式：max_blocks = 5 + 20 * (1 - exp(-avg_size/600))
                    max_blocks = 5 + 20 * (1 - math.exp(-avg_size / 600))
                    max_blocks = max(5, min(25, int(max_blocks)))  # 限制在 5-25 之间

                    scaled_blocks = max(3, min(max_blocks, calculated_blocks))

                    # 如果区域很小，确保至少有 3 块
                    if avg_size < 30:
                        scaled_blocks = max(3, scaled_blocks)

                    print(f"[Japan Mode] Region size: {w_expanded}x{h_expanded} (avg={avg_size:.1f}px), target_block_size={target_block_size:.1f}px, max_blocks={max_blocks}, Calculated blocks: {scaled_blocks}")
                elif block_count_scaling != "fixed":
                    d_pct = max(h_expanded / image_height, w_expanded / image_width)
                    if block_count_scaling == "fewer_when_large":
                        # 区域越大，块数越少（从 blocks 向 min_blocks 过渡）。
                        # 确保大区域时块数不会太多，保持打码效果
                        min_blocks = max(1, min(5, blocks // 4))  # 大区域时最少块数（但不超过 blocks/4）
                        scaled_blocks = int(blocks + d_pct * (min_blocks - blocks))
                        scaled_blocks = max(1, min(scaled_blocks, blocks))  # 限制在 [1, blocks] 范围内
                    else:  # elif block_count_scaling == "fewer_when_small"
                        # 区域越小，块数越少（从 1 向 blocks 过渡）。
                        scaled_blocks = int(1 + d_pct * (blocks - 1))
                        scaled_blocks = max(1, min(scaled_blocks, blocks))  # 限制在 [1, blocks] 范围内
                else:
                    # fixed 模式：直接使用 blocks，但如果 blocks 太大，给出警告或限制
                    # 为了保持打码效果，建议 blocks <= 10，但这里不强制限制，让用户自己控制
                    scaled_blocks = blocks

                # 使用像素化处理目标区域，传入 corner_ratio 参数
                # corner_ratio > 0 时启用圆角效果
                rounded = corner_ratio > 0.0
                image[y_expanded: y_expanded + h_expanded, x_expanded: x_expanded + w_expanded] = pixelate(
                    area,
                    blocks=scaled_blocks,
                    rounded=rounded,
                    corner_ratio=corner_ratio
                )

        # 将 numpy 图像转回 torch，并恢复到批次维度。
        output_images.append(torch.from_numpy(image).unsqueeze(0))

    # 拼接回 [B, H, W, C] 的批次输出。
    return torch.cat(output_images, dim=0)


class ApplyNudenet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nudenet_model": ("NUDENET_MODEL",),
                "image": ("IMAGE",),
                "filtered_labels": ("FILTERED_LABELS",),
                "min_score": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "blocks": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "block_count_scaling": (BLOCK_COUNT_BEHAVIOURS, {"tooltip": "Scale block count by censor area."}),
                "inference_resolution": (
                    "INT",
                    {
                        "default": 320,
                        "min": 320,
                        "max": 640,
                        "step": 320,
                        "tooltip": "推理分辨率：320=320n模型（快速，默认），640=640m模型（更精确但较慢）。原版NudeNet支持320n和640m两种模型。",
                    },
                ),
                "corner_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "圆角程度：0=矩形（默认），越大越接近椭圆。建议范围 0.0-0.5。",
                    },
                ),
                "expand_pixels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "扩展马赛克区域的像素数，用于扩大打码范围。例如设置为10，会在检测框四周各扩展10像素。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_nudenet"
    CATEGORY = "Nudenet"

    def apply_nudenet(
        self,
        nudenet_model,
        image,
        filtered_labels,
        min_score,
        blocks,
        block_count_scaling,
        inference_resolution,
        corner_ratio,
        expand_pixels,
    ):
        # 默认使用 320（320n 模型），可选择 640（640m 模型，更精确但较慢）
        output_image = nudenet_execute(
            nudenet_model=nudenet_model,
            input_images=image,
            filtered_labels=filtered_labels,
            min_score=min_score,
            blocks=blocks,
            block_count_scaling=block_count_scaling,
            inference_resolution=inference_resolution,
            corner_ratio=corner_ratio,
            expand_pixels=expand_pixels,
        )
        return (output_image,)


class FilteredLabel:

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types dynamically based on CLASSIDS_LABELS_MAPPING.
        使用标签名作为输入（与原版一致）。

        Returns:
            dict: A dictionary of required input types with default values.
        """
        return {
            "required": {
                key: ("BOOLEAN", {"default": False})
                for key in LABELS_CLASSIDS_MAPPING.keys()  # 使用标签名作为 key
            }
        }

    RETURN_TYPES = ("FILTERED_LABELS",)
    FUNCTION = "filter_labels"
    CATEGORY = "Nudenet"

    def filter_labels(self, *args, **kwags):
        # 返回需要打码的标签名列表
        # 只有勾选（True）的标签才会被打码，默认都是 False（不打码）
        censored_labels = []
        for label, should_censor in kwags.items():
            if should_censor:  # 如果勾选（True），则加入打码列表
                censored_labels.append(label)
        return (censored_labels,)


class NudenetModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("Nudenet"),),
            }
        }

    RETURN_TYPES = ("NUDENET_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nudenet"

    def load_model(self, model, providers="CPU"):
        import onnxruntime
        from onnxruntime.capi import _pybind_state as C

        # 优先使用 GPU（如果可用），否则使用 CPU
        # 原版默认使用 CPU（providers 参数被注释），这里优化为自动选择最佳 provider
        available_providers = C.get_available_providers()

        # 如果可用，优先使用 CUDA（GPU）
        if "CUDAExecutionProvider" in available_providers:
            selected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[NudenetModelLoader] Using GPU (CUDA) for faster inference")
        else:
            selected_providers = ["CPUExecutionProvider"]
            print(f"[NudenetModelLoader] Using CPU (GPU not available, install onnxruntime-gpu for GPU support)")

        self.session = onnxruntime.InferenceSession(
            os.path.join(current_paths[0], model),
            providers=selected_providers
        )

        # 显示实际使用的 provider
        actual_providers = self.session.get_providers()
        print(f"[NudenetModelLoader] Model loaded with providers: {actual_providers}")

        model_inputs = self.session.get_inputs()
        self.input_width = model_inputs[0].shape[2]
        self.input_height = model_inputs[0].shape[3]
        self.input_name = model_inputs[0].name

        return (
            ModelLoader(
                session=self.session,
                input_width=self.input_width,
                input_height=self.input_height,
                input_name=self.input_name,
            ),
        )


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Register
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "ApplyNudenet_new": ApplyNudenet,
    # Loaders
    "NudenetModelLoader_new": NudenetModelLoader,
    # Helpers
    "FilterdLabel_new": FilteredLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "ApplyNudenet_new": "Apply Nudenet NEW",
    # Loaders
    "NudenetModelLoader_new": "Nudenet Model Loader NEW",
    # Helpers
    "FilterdLabel_new": "Filtered Labels NEW",
}
