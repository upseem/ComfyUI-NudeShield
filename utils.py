
import numpy as np
import cv2

def pixelate(
    image: np.ndarray,
    blocks: int = 3,
    rounded: bool = False,
    corner_ratio: float = 0.25,
    smooth: int = 0,
) -> np.ndarray:
    """将图像做"像素化"处理（马赛克效果）。

    工作原理:
        1. 按给定 blocks 数将区域在宽、高方向分别均匀切成 blocks 份，形成 blocks×blocks 网格。
        2. 遍历每个网格块，计算该块所有像素的均值颜色（对 H×W×C 的 ROI 在轴 (0,1) 上做 mean）。
        3. 使用该均值填充整个网格块，降低细节，形成像素化视觉效果。
        4. 如果启用圆角，对整个区域（而非每个小块）应用圆角遮罩。

    参数:
        image (np.ndarray): 输入图像，形状通常为 (H, W, C)。该函数会原地修改此数组。
        blocks (int): 网格块数量（行列相同）。越大 -> 单块越小 -> 细节保留稍多；越小 -> 像素化越粗糙。
        rounded (bool): 是否对整个区域应用圆角（而非每个小块）。默认 False（保持原矩形区域）。
        corner_ratio (float): 圆角半径相对于区域短边的比例，建议 [0.0, 0.5]。例如 0.25 表示圆角半径≈min(w, h)*0.25。
        smooth (int): 边缘平滑（羽化）强度，表示用于平滑圆角边缘的高斯核大小（像素，必须为非负整数；>0 时自动取奇数）。

    返回:
        np.ndarray: 处理后的同一个数组（与输入对象引用一致）。
    """
    (h, w) = image.shape[:2]  # 只需要高度和宽度；通道数不参与网格划分。

    # 如果未启用圆角，使用与 pixelate_old 完全相同的逻辑
    if not rounded:
        # 完全复制 pixelate_old 的逻辑，确保颜色一致
        x_steps = np.linspace(0, w, blocks + 1, dtype="int")
        y_steps = np.linspace(0, h, blocks + 1, dtype="int")

        for i in range(1, len(y_steps)):
            for j in range(1, len(x_steps)):
                start_x = x_steps[j - 1]
                start_y = y_steps[i - 1]
                end_x = x_steps[j]
                end_y = y_steps[i]

                roi = image[start_y:end_y, start_x:end_x]
                mean_color = np.mean(roi, axis=(0, 1))
                image[start_y:end_y, start_x:end_x] = mean_color

        return image

    # 启用圆角时，需要保存原始图像
    original_image = image.copy()

    # 生成网格切分点：blocks+1 个边界（含 0 和 w/h）。使用 linspace 保证均匀分割。
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    # 遍历每个网格块（跳过第 0 个边界，所以从 1 开始）。
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            # 取出该块的 ROI；形状 (block_h, block_w, C)。
            roi = image[start_y:end_y, start_x:end_x]
            # 对该块做颜色均值；axis=(0,1) 表示在空间维度上求平均，保留通道维度。
            mean_color = np.mean(roi, axis=(0, 1))
            # 用均值填满 ROI（先进行像素化，圆角稍后处理）
            image[start_y:end_y, start_x:end_x] = mean_color

    # 如果启用圆角，对整个区域应用圆角遮罩
    if rounded:
        # 圆角半径（像素）：基于整个区域短边计算
        r_max = 0.5 * min(w, h)
        r = float(corner_ratio) * min(w, h)
        r = max(0.0, min(r, r_max))

        # 若 r 过小，相当于矩形区域；不需要处理
        if r >= 1.0:
            # 生成像素坐标网格（以左上角为原点）
            ys = np.arange(h, dtype=np.float32)[:, None]
            xs = np.arange(w, dtype=np.float32)[None, :]

            # 核心矩形区域（无需圆角处）
            in_core_x = (xs >= r) & (xs <= (w - 1 - r))
            in_core_y = (ys >= r) & (ys <= (h - 1 - r))
            inside_core = in_core_x | in_core_y

            # 四角的四分之一圆：以 (r, r)、(w-1-r, r)、(r, h-1-r)、(w-1-r, h-1-r) 为圆心
            tl = (xs - r) ** 2 + (ys - r) ** 2 <= r ** 2
            tr = (xs - (w - 1 - r)) ** 2 + (ys - r) ** 2 <= r ** 2
            bl = (xs - r) ** 2 + (ys - (h - 1 - r)) ** 2 <= r ** 2
            br = (xs - (w - 1 - r)) ** 2 + (ys - (h - 1 - r)) ** 2 <= r ** 2

            inside = inside_core | tl | tr | bl | br
            mask = inside.astype(np.float32)

            # 边缘平滑：对 mask 做高斯模糊，获得软边（值域近似 [0,1]）。
            if smooth and smooth > 0:
                k = int(smooth)
                if k % 2 == 0:
                    k += 1  # 高斯核大小需为奇数
                # 保护极小块：核不超过区域尺寸且 >=3
                k = max(3, min(k, h | 1, w | 1))
                mask = cv2.GaussianBlur(mask, (k, k), 0)
                mask = np.clip(mask, 0.0, 1.0)

            # 融合：image = original*(1-mask) + pixelated*mask
            # 圆角区域外恢复原图，圆角区域内保持像素化
            # 确保数据类型正确，避免颜色变浅
            if image.dtype == np.uint8:
                # 对于 uint8，先转换为 float 进行计算，再转回
                image_f = image.astype(np.float32)
                original_f = original_image.astype(np.float32)
                blended = original_f * (1.0 - mask[:, :, None]) + image_f * mask[:, :, None]
                # 使用 np.round 确保颜色深度正确
                blended = np.round(blended).clip(0, 255).astype(np.uint8)
                image[:] = blended
            else:
                # 对于 float 类型，直接计算
                image_f = image.astype(np.float32) if image.dtype != np.float32 else image
                original_f = original_image.astype(np.float32) if original_image.dtype != np.float32 else original_image
                blended = original_f * (1.0 - mask[:, :, None]) + image_f * mask[:, :, None]
                if image.dtype != np.float32:
                    blended = blended.astype(image.dtype)
                image[:] = blended

    return image


def pixelate_old(image: np.ndarray, blocks: int = 3) -> np.ndarray:
    (h, w) = image.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            roi = image[start_y:end_y, start_x:end_x]
            mean_color = np.mean(roi, axis=(0, 1))
            image[start_y:end_y, start_x:end_x] = mean_color

    return image
