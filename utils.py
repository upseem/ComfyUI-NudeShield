
import numpy as np
import cv2





def pixelate(
    image: np.ndarray,
    blocks: int = 3,
    rounded: bool = False,
    corner_ratio: float = 0.25,
    smooth: int = 0,
) -> np.ndarray:
    """将图像做“像素化”处理（马赛克效果）。

    工作原理:
        1. 按给定 blocks 数将区域在宽、高方向分别均匀切成 blocks 份，形成 blocks×blocks 网格。
        2. 遍历每个网格块，计算该块所有像素的均值颜色（对 H×W×C 的 ROI 在轴 (0,1) 上做 mean）。
        3. 使用该均值填充整个网格块，降低细节，形成像素化视觉效果。

    参数:
        image (np.ndarray): 输入图像，形状通常为 (H, W, C)。该函数会原地修改此数组。
        blocks (int): 网格块数量（行列相同）。越大 -> 单块越小 -> 细节保留稍多；越小 -> 像素化越粗糙。
        rounded (bool): 是否使用圆角像素块（将矩形块替换为“圆角矩形/椭圆”形状的填充）。默认 False（保持原矩形块）。
        corner_ratio (float): 圆角半径相对于单块短边的比例，建议 [0.0, 0.5]。例如 0.25 表示圆角半径≈min(block_w, block_h)*0.25。
                              当块为非正方形时，视觉上会更像“椭圆圆角”。过大时形状接近椭圆。
        smooth (int): 边缘平滑（羽化）强度，表示用于平滑圆角边缘的高斯核大小（像素，必须为非负整数；>0 时自动取奇数）。

    返回:
        np.ndarray: 处理后的同一个数组（与输入对象引用一致）。

    复杂度:
        时间复杂度约 O(H*W + blocks^2 * avg_block_pixels)，主要开销在均值计算；
        空间复杂度 O(1)，只使用少量临时变量，不分配新图像。

    注意:
        - blocks 最小建议为 1（=不做任何网格切分，循环仍会赋值为自身均值）。
        - 输入 dtype 若为浮点或整数均可，均值结果会是浮点；直接赋值可能导致类型提升或截断（由 numpy 广播与数组 dtype 决定）。
                - 函数是“原地”修改图像，如需保留原图请先复制：`image.copy()`。
                - 若开启 rounded，单块内部将以圆角形状进行加权合成（原像素与均值颜色按掩码融合），
                    块四角处会保留一定原始像素（不会出现锯齿边），在相邻块交界处因此可能保留原图细节，从而得到更“柔和”的像素化观感。
    """
    (h, w) = image.shape[:2]  # 只需要高度和宽度；通道数不参与网格划分。

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

            if not rounded:
                # 传统像素化：用均值填满 ROI。
                image[start_y:end_y, start_x:end_x] = mean_color
            else:
                # 圆角像素化：构建圆角掩码，仅在圆角区域内用 mean_color 替换，边缘可选平滑。
                block_h = max(1, end_y - start_y)
                block_w = max(1, end_x - start_x)

                # 圆角半径（像素）：基于单块短边计算，并限制到 [0, min(block_w, block_h)/2]
                r_max = 0.5 * min(block_w, block_h)
                r = float(corner_ratio) * min(block_w, block_h)
                r = max(0.0, min(r, r_max))

                # 若 r 过小，相当于矩形块；直接填充提升效率
                if r < 1.0:
                    image[start_y:end_y, start_x:end_x] = mean_color
                    continue

                # 生成像素坐标网格（以左上角为原点）
                ys = np.arange(block_h, dtype=np.float32)[:, None]
                xs = np.arange(block_w, dtype=np.float32)[None, :]

                # 核心矩形区域（无需圆角处）
                in_core_x = (xs >= r) & (xs <= (block_w - 1 - r))
                in_core_y = (ys >= r) & (ys <= (block_h - 1 - r))
                inside_core = in_core_x | in_core_y

                # 四角的四分之一圆：以 (r, r)、(w-1-r, r)、(r, h-1-r)、(w-1-r, h-1-r) 为圆心
                tl = (xs - r) ** 2 + (ys - r) ** 2 <= r ** 2
                tr = (xs - (block_w - 1 - r)) ** 2 + (ys - r) ** 2 <= r ** 2
                bl = (xs - r) ** 2 + (ys - (block_h - 1 - r)) ** 2 <= r ** 2
                br = (xs - (block_w - 1 - r)) ** 2 + (ys - (block_h - 1 - r)) ** 2 <= r ** 2

                inside = inside_core | tl | tr | bl | br
                mask = inside.astype(np.float32)

                # 边缘平滑：对 mask 做高斯模糊，获得软边（值域近似 [0,1]）。
                if smooth and smooth > 0:
                    k = int(smooth)
                    if k % 2 == 0:
                        k += 1  # 高斯核大小需为奇数
                    # 保护极小块：核不超过块尺寸且 >=3
                    k = max(3, min(k, block_h | 1, block_w | 1))
                    mask = cv2.GaussianBlur(mask, (k, k), 0)
                    mask = np.clip(mask, 0.0, 1.0)

                # 融合：roi = roi*(1-mask) + mean_color*mask
                if roi.dtype != np.float32:
                    roi_f = roi.astype(np.float32)
                else:
                    roi_f = roi
                mean_color_vec = mean_color.reshape(1, 1, -1).astype(np.float32)
                blended = roi_f * (1.0 - mask[:, :, None]) + mean_color_vec * mask[:, :, None]

                # 写回，保持原 dtype
                if roi.dtype != np.float32:
                    blended = blended.astype(roi.dtype)
                image[start_y:end_y, start_x:end_x] = blended

    return image
