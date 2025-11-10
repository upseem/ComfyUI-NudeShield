from ultralytics import YOLO
from nudenet import NudeDetector
from PIL import Image
import numpy as np
import cv2
import os


def nsfw_censor_image(
    image_path: str,
    save_dir: str = "outputs/nsfw_mask",
    conf: float = 0.1,
    imgsz: int = 832,
    mosaic_block: int = 25,
    mask_expand: float = 0.2,
):
    """
    基于 YOLO NSFW 模型检测敏感部位并在掩膜区域精准打码。

    Args:
        image_path: 输入图片路径
        save_dir: 输出文件夹
        conf: 置信度阈值 (默认 0.1)
        imgsz: 模型推理分辨率 (默认 832)
        mosaic_block: 马赛克块数量（数量越多越清晰）
        mask_expand: 区域扩张比例 (0.2 表示上下左右各扩 20%)
    Returns:
        (mask_path, censored_path)
    """

    os.makedirs(save_dir, exist_ok=True)

    # 加载三个模型
    models = [
        YOLO("nsfw-seg-breast-x.pt"),
        YOLO("nsfw-seg-penis-x.pt"),
        YOLO("nsfw-seg-vagina-x.pt"),
    ]

    # 读取图片 - 使用与base()函数相同的方式
    print(f"📷 读取图片: {image_path}")
    
    # 直接使用文件路径，避免PIL转换问题
    # 这样与base()函数保持一致
    try:
        # 先检查图片信息
        img_pil = Image.open(image_path)
        print(f"📐 原始图片尺寸: {img_pil.size}, 模式: {img_pil.mode}")
        
        # 获取图片尺寸用于后续处理
        h, w = img_pil.size[1], img_pil.size[0]  # PIL的size是(width, height)
        print(f"📐 图片尺寸: {w}x{h}")
        
    except Exception as e:
        print(f"❌ 图片信息读取失败: {e}")
        return None, None

    # 初始化总掩膜
    final_mask = np.zeros((h, w), np.uint8)

    # 推理并合并掩膜 - 使用直接文件路径，避免转换问题
    detection_count = 0
    for i, model in enumerate(models):
        print(f"🔍 使用模型 {i+1}/3 进行检测...")
        # 直接使用文件路径，与base()函数保持一致
        results = model.predict(image_path, imgsz=imgsz, conf=conf, verbose=False)
        
        # 检查是否有检测结果
        if results[0].masks is not None and len(results[0].masks) > 0:
            print(f"✅ 模型 {i+1} 检测到 {len(results[0].masks)} 个区域")
            detection_count += len(results[0].masks)
            
        masks = results[0].masks.data.cpu().numpy()
        for mask in masks:
                # 确保mask是二值化的
            mask = (mask > 0.5).astype(np.uint8)
                # 调整mask尺寸到原图尺寸
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # 区域扩张
            if mask_expand > 0:
                k = int(max(h, w) * mask_expand / 10)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # 合并到总掩膜
            final_mask = cv2.bitwise_or(final_mask, mask)
        else:
            print(f"❌ 模型 {i+1} 未检测到任何区域")
    
    print(f"📊 总共检测到 {detection_count} 个敏感区域")

    mask_path = os.path.join(save_dir, "mask_total.png")
    cv2.imwrite(mask_path, final_mask * 255)
    print(f"✅ 掩膜已保存: {mask_path}")

    # 未检测到任何区域
    if not np.any(final_mask):
        print("⚠️ 未检测到 NSFW 区域。")
        return mask_path, None

    # 重新读取图片进行马赛克处理
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    # 生成整图马赛克 - 修改逻辑：mosaic_block表示马赛克块数量，数量越多越清晰
    h_small = max(1, mosaic_block)
    w_small = max(1, mosaic_block)
    small = cv2.resize(img_np, (w_small, h_small),
                       interpolation=cv2.INTER_LINEAR)
    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 添加模糊处理，让轮廓更柔和
    if mosaic_block <= 16:  # 当块数量较少时，添加模糊
        blur_radius = max(3, min(15, 32 // mosaic_block))  # 根据块数量调整模糊半径
        # 确保核大小是正奇数
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        mosaic_full = cv2.GaussianBlur(mosaic_full, (blur_radius, blur_radius), 0)

    # 精确贴合掩膜区域
    img_censored = img_np.copy()
    img_censored[final_mask == 1] = mosaic_full[final_mask == 1]

    censored_path = os.path.join(save_dir, "censored_result.jpg")
    cv2.imwrite(censored_path, cv2.cvtColor(img_censored, cv2.COLOR_RGB2BGR))
    print(f"✅ 精准马赛克图片已保存: {censored_path}")

    return mask_path, censored_path


all_labels = [
    "FEMALE_GENITALIA_COVERED",  # 女性生殖部位（被遮挡，例如穿内裤或衣物覆盖）
    "FACE_FEMALE",               # 女性人脸（检测女性面部）
    "BUTTOCKS_EXPOSED",          # 臀部暴露（未穿衣物或部分裸露）
    "FEMALE_BREAST_EXPOSED",     # 女性胸部暴露（裸露或低胸）
    "FEMALE_GENITALIA_EXPOSED",  # 女性生殖部位暴露（阴部裸露）
    "MALE_BREAST_EXPOSED",       # 男性胸部暴露（通常为赤膊上身）
    "ANUS_EXPOSED",              # 肛门暴露
    "FEET_EXPOSED",              # 脚部暴露（赤脚或脚裸露）
    "BELLY_COVERED",             # 腹部被覆盖（衣物遮挡但能识别该区域）
    "FEET_COVERED",              # 脚部被遮挡（穿鞋或袜）
    "ARMPITS_COVERED",           # 腋下被覆盖（穿衣物）
    "ARMPITS_EXPOSED",           # 腋下暴露（无袖衣物或举手动作导致暴露）
    "FACE_MALE",                 # 男性人脸
    "BELLY_EXPOSED",             # 腹部暴露（例如短上衣或裸露上身）
    "MALE_GENITALIA_EXPOSED",    # 男性生殖器暴露（阴茎或阴囊裸露）
    "ANUS_COVERED",              # 肛门区域被衣物覆盖（识别为臀部但非裸露）
    "FEMALE_BREAST_COVERED",     # 女性胸部被遮挡（穿衣物、内衣等）
    "BUTTOCKS_COVERED",          # 臀部被遮挡（穿裤子、裙子等）
]


def apply_japanese_style_mosaic(img_np, mask, mosaic_block=8):
    """
    应用日本AV风格的马赛克处理
    
    Args:
        img_np: 输入图像数组
        mask: 掩膜区域
        mosaic_block: 马赛克块数量
    
    Returns:
        处理后的图像数组
    """
    h, w = img_np.shape[:2]
    
    # 生成马赛克
    h_small = max(1, mosaic_block)
    w_small = max(1, mosaic_block)
    small = cv2.resize(img_np, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 日本风格处理：强模糊 + 颜色统一
    if mosaic_block <= 50:
        # 1. 高斯模糊 - 更强的模糊效果
        blur_radius = max(8, min(30, 80 // mosaic_block))
        # 确保核大小是正奇数
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        mosaic_full = cv2.GaussianBlur(mosaic_full, (blur_radius, blur_radius), 0)
        
        # 2. 双边滤波 - 颜色统一，减少细节
        mosaic_full = cv2.bilateralFilter(mosaic_full, 20, 100, 100)
        
        # 3. 额外的平滑处理
        mosaic_full = cv2.medianBlur(mosaic_full, 5)
        
        print(f"🇯🇵 日本风格处理: 模糊半径 {blur_radius}, 双边滤波, 中值滤波")
    
    # 应用马赛克到掩膜区域
    result = img_np.copy()
    result[mask > 0] = mosaic_full[mask > 0]
    
    return result


def nsfw_censor_image_nudenet(
    image_path: str,
    output_path: str = None,
    conf: float = 0.5,
    model_path: str = None,
    inference_resolution: int = 320,
    labels: list = None,
    mosaic_block: int = 8,
    mask_expand: float = 0,
    japanese_style: bool = False
):
    """
    使用NudeNet对单张图片进行NSFW检测和打码处理
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径，如果为None则自动生成
        conf: 置信度阈值
        model_path: 模型路径，如果为None则使用默认模型
        inference_resolution: 推理分辨率
        labels: 要检测的标签列表，如果为None则使用所有标签
        mosaic_block: 马赛克块数量（数量越多越清晰）
        mask_expand: 掩膜扩张比例
        japanese_style: 是否使用日本AV风格的马赛克处理
    
    Returns:
        tuple: (成功标志, 输出路径)
    """
    print(f"🖼️ 开始处理图片: {image_path}")
    
    # 检查输入文件
    if not os.path.exists(image_path):
        print(f"❌ 输入文件不存在: {image_path}")
        return False, None
    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_censored.jpg")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载NudeNet模型
        print("🤖 加载NudeNet检测模型...")
        if model_path and os.path.exists(model_path):
            detector = NudeDetector(model_path, inference_resolution=inference_resolution)
            print(f"📦 使用自定义模型: {os.path.basename(model_path)} (分辨率: {inference_resolution})")
        else:
            detector = NudeDetector(inference_resolution=inference_resolution)
            print(f"📦 使用默认320n模型 (分辨率: {inference_resolution})")
        
        # 2. 设置检测标签
        if labels is None:
            target_labels = all_labels
            print("🎯 使用所有标签进行检测")
        else:
            target_labels = labels
            print(f"🎯 使用指定标签进行检测: {target_labels}")
        
        # 3. 进行检测
        print("🔍 开始检测...")
        detections = detector.detect(image_path)
        
        # 过滤低置信度和指定标签的检测
        filtered_detections = [
            d for d in detections 
            if d['score'] >= conf and d['class'] in target_labels
        ]
        
        print(f"📊 检测结果: 共检测到 {len(detections)} 个区域，过滤后 {len(filtered_detections)} 个")
        
        # 4. 读取图片
        img_pil = Image.open(image_path)
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        print(f"📐 图片尺寸: {w}x{h}")
        
        # 5. 创建掩膜
        current_mask = np.zeros((h, w), dtype=np.uint8)
        
        if filtered_detections:
            print("🎨 创建掩膜...")
            for detection in filtered_detections:
                # NudeNet坐标格式: [x, y, width, height]
                x, y, width, height = detection['box']
                print(f"🔍 检测区域: {detection['class']} 置信度: {detection['score']:.3f} 坐标: [x={x}, y={y}, w={width}, h={height}]")
                
                # 转换为边界坐标
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)
                
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 确保坐标有效
                if x2 > x1 and y2 > y1:
                    # 创建矩形掩膜
                    current_mask[y1:y2, x1:x2] = 255
                    print(f"✅ 掩膜区域: [{x1}, {y1}, {x2}, {y2}]")
                else:
                    print(f"⚠️ 无效坐标: [{x1}, {y1}, {x2}, {y2}]")
        
        # 6. 区域扩张
        final_mask = current_mask.copy()
        if mask_expand > 0 and np.any(final_mask > 0):
            k = int(max(h, w) * mask_expand / 10)
            if k > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                final_mask = cv2.dilate(final_mask, kernel, iterations=1)
                print(f"🔧 掩膜扩张: 核大小 {k}x{k}")
        
        print(f"📊 掩膜统计: 总像素 {h*w}, 掩膜像素 {np.sum(final_mask > 0)}")
        
        # 7. 应用马赛克
        if np.any(final_mask > 0):
            print("🎨 生成马赛克...")
            
            if japanese_style:
                # 使用日本风格处理
                print(f"🇯🇵 使用日本AV风格马赛克处理")
                img_np = apply_japanese_style_mosaic(img_np, final_mask, mosaic_block)
                print(f"✅ 日本风格马赛克应用成功: {np.sum(final_mask > 0)} 个像素被替换")
            else:
                # 标准处理
                h_small = max(1, mosaic_block)
                w_small = max(1, mosaic_block)
                small = cv2.resize(img_np, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                print(f"🎨 马赛克生成: 原图 {w}x{h} -> 小图 {w_small}x{h_small} -> 马赛克 {w}x{h} (块数量: {mosaic_block}x{mosaic_block})")
                
                # 添加模糊处理，让轮廓更柔和
                if mosaic_block <= 50:  # 当块数量较少时，添加模糊
                    blur_radius = max(3, min(15, 32 // mosaic_block))  # 根据块数量调整模糊半径
                    # 确保核大小是正奇数
                    blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                    mosaic_full = cv2.GaussianBlur(mosaic_full, (blur_radius, blur_radius), 0)
                    print(f"🔍 应用模糊处理: 模糊半径 {blur_radius}x{blur_radius}")
                
                # 应用马赛克到检测区域
                img_np[final_mask > 0] = mosaic_full[final_mask > 0]
                print(f"✅ 马赛克应用成功: {np.sum(final_mask > 0)} 个像素被替换")
        else:
            print("📋 未检测到NSFW内容，直接复制原图")
        
        # 8. 保存处理后的图片
        result_img = Image.fromarray(img_np)
        
        # 根据输出路径的扩展名确定保存格式
        output_ext = os.path.splitext(output_path)[1].lower()
        if output_ext in ['.jpg', '.jpeg']:
            result_img.save(output_path, 'JPEG', quality=95)
        elif output_ext == '.png':
            result_img.save(output_path, 'PNG')
        elif output_ext == '.webp':
            result_img.save(output_path, 'WEBP', quality=95)
        else:
            # 默认保存为JPEG格式
            if not output_ext:
                output_path += '.jpg'
            result_img.save(output_path, 'JPEG', quality=95)
        
        print(f"💾 保存处理后的图片: {output_path}")
        
        print("✅ 图片处理完成!")
        return True, output_path
        
    except Exception as e:
        print(f"❌ 图片处理失败: {e}")
        return False, None


def nsfw_censor_video_nudenet(
    video_path: str,
    output_path: str = None,
    conf: float = 0.5,
    model_path: str = None,
    inference_resolution: int = 320,
    temp_dir: str = "temp_frames_nudenet",
    labels: list = None,
    mosaic_block: int = 8,
    mask_expand: float = 0
):
    """
    基于NudeNet的NSFW视频检测和打码功能
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径，默认为原文件名_censored.mp4
        conf: 置信度阈值
        model_path: NudeNet模型路径，None使用默认320n模型
        inference_resolution: 推理分辨率
        temp_dir: 临时帧目录
        labels: 要检测的标签列表，None使用所有标签
        mosaic_block: 马赛克块数量（数量越多越清晰）
        mask_expand: 掩膜扩张比例
    
    Returns:
        tuple: (是否成功, 输出视频路径)
    """
    try:
        from nudenet import NudeDetector
    except ImportError:
        print("❌ 请先安装NudeNet: pip install --upgrade 'nudenet>=3.4.2'")
        return False, None

    import subprocess
    import shutil
    import glob

    print(f"🎬 开始处理视频: {video_path}")

    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_censored_nudenet.mp4"

    # 确保输出路径有正确的扩展名
    if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        output_path += '.mp4'

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")

    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    frames_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    detection_dir = os.path.join(temp_dir, "detections")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(detection_dir, exist_ok=True)

    try:
        # 1. 使用FFmpeg提取视频信息
        print("📹 获取视频信息...")
        cmd_info = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd_info, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ 无法获取视频信息: {result.stderr}")
            return False, None

        import json
        video_info = json.loads(result.stdout)
        video_stream = None
        audio_stream = None

        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
            elif stream['codec_type'] == 'audio':
                audio_stream = stream

        if not video_stream:
            print("❌ 未找到视频流")
            return False, None

        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # 尝试获取帧率，如果失败则使用默认30fps
        try:
            fps = eval(video_stream['r_frame_rate'])  # 计算帧率
            if fps <= 0 or fps > 60:  # 帧率异常时使用默认值
                fps = 30.0
                print(f"⚠️ 检测到异常帧率，使用默认30fps")
        except (ValueError, ZeroDivisionError, KeyError):
            fps = 30.0
            print(f"⚠️ 无法获取帧率，使用默认30fps")

        print(f"📹 视频信息: {width}x{height}, {fps:.2f}fps")
        if audio_stream:
            print(f"🎵 音频流: {audio_stream['codec_name']}")

        # 2. 提取所有帧
        print("🎬 提取视频帧...")
        cmd_extract = [
            "ffmpeg", "-i", video_path,  # 不添加fps滤镜，提取所有原始帧
            os.path.join(frames_dir, "frame_%06d.png")
        ]
        result = subprocess.run(cmd_extract, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ 帧提取失败: {result.stderr}")
            return False, None

        # 获取帧文件列表
        frame_files = sorted(
            glob.glob(os.path.join(frames_dir, "frame_*.png")))
        total_frames = len(frame_files)
        print(f"📊 提取了 {total_frames} 帧")

        if total_frames == 0:
            print("❌ 未提取到任何帧")
            return False, None

        # 3. 加载NudeNet模型
        print("🤖 加载NudeNet检测模型...")
        if model_path:
            detector = NudeDetector(
                model_path=model_path, inference_resolution=inference_resolution)
            print(f"📦 使用自定义模型: {model_path}")
        else:
            detector = NudeDetector()
            print("📦 使用默认320n模型")

        # 设置检测标签
        if labels is None:
            target_labels = all_labels
            print(f"🎯 使用所有标签进行检测: {len(target_labels)} 个类别")
        else:
            target_labels = labels
            print(f"🎯 使用指定标签进行检测: {target_labels}")

        # 4. 逐帧处理
        print("🔍 开始逐帧检测和处理...")
        nsfw_frames = 0

        for i, frame_file in enumerate(frame_files):
            if (i + 1) % 30 == 0:  # 每30帧显示进度
                progress = ((i + 1) / total_frames) * 100
                print(f"📊 处理进度: {i+1}/{total_frames} ({progress:.1f}%)")

            # 使用NudeNet检测
            try:
                detections = detector.detect(frame_file)

                # 过滤低置信度和指定标签的检测
                filtered_detections = [
                    d for d in detections
                    if d['score'] >= conf and d['class'] in target_labels
                ]

                # 保存检测结果
                detection_result = {
                    'frame': i + 1,
                    'detections': filtered_detections,
                    'count': len(filtered_detections),
                    'target_labels': target_labels
                }

                detection_file = os.path.join(
                    detection_dir, f"frame_{i+1:06d}.json")
                with open(detection_file, 'w', encoding='utf-8') as f:
                    json.dump(detection_result, f,
                              ensure_ascii=False, indent=2)

                # 打印检测结果
                if filtered_detections:
                    detection_classes = [d['class']
                                         for d in filtered_detections]
                    # print(f"📊 帧 {i+1:06d}: 检测到 {len(filtered_detections)} 个区域: {detection_classes}")
                else:
                    print(f"📊 帧 {i+1:06d}: 未检测到指定标签的NSFW内容")

                # 如果有检测到NSFW内容，进行马赛克处理
                if filtered_detections:
                    nsfw_frames += 1

                    # 读取原始帧
                    img_pil = Image.open(frame_file)
                    img_np = np.array(img_pil)
                    h, w = img_np.shape[:2]

                    # 创建掩膜
                    mask = np.zeros((h, w), dtype=np.uint8)

                    # 为每个检测区域创建掩膜
                    for detection in filtered_detections:
                        # NudeNet坐标格式: [x, y, width, height]
                        x, y, width, height = detection['box']
                        print(
                            f"📊 帧 {i+1:06d} 检测区域: {detection['class']} 置信度: {detection['score']:.3f} 坐标: [x={x}, y={y}, w={width}, h={height}]")

                        # 转换为边界坐标
                        x1, y1 = int(x), int(y)
                        x2, y2 = int(x + width), int(y + height)

                        # 确保坐标在图像范围内
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # print(f"🔧 转换后坐标: [{x1}, {y1}, {x2}, {y2}]")

                        # 确保坐标有效
                        if x2 > x1 and y2 > y1:
                            # 创建矩形掩膜
                            mask[y1:y2, x1:x2] = 255
                            # print(f"✅ 创建掩膜区域: [{x1}, {y1}, {x2}, {y2}] 大小: {x2-x1}x{y2-y1}")
                        else:
                            print(f"⚠️ 无效坐标: [{x1}, {y1}, {x2}, {y2}]")

                    # 区域扩张（在所有检测区域创建后统一扩张）
                    if mask_expand > 0 and np.any(mask > 0):
                        k = int(max(h, w) * mask_expand / 10)
                        if k > 0:
                            kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (k, k))
                            mask = cv2.dilate(mask, kernel, iterations=1)
                            print(f"🔧 掩膜扩张: 核大小 {k}x{k}")

                    # print(f"📊 掩膜统计: 总像素 {h*w}, 掩膜像素 {np.sum(mask > 0)}")

                    # 生成马赛克 - 修改逻辑：mosaic_block表示马赛克块数量，数量越多越清晰
                    h_small = max(1, mosaic_block)
                    w_small = max(1, mosaic_block)
                    small = cv2.resize(
                        img_np, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                    mosaic_full = cv2.resize(
                        small, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # 添加模糊处理，让轮廓更柔和
                    if mosaic_block <= 16:  # 当块数量较少时，添加模糊
                        blur_radius = max(3, min(15, 32 // mosaic_block))  # 根据块数量调整模糊半径
                        # 确保核大小是正奇数
                        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                        mosaic_full = cv2.GaussianBlur(mosaic_full, (blur_radius, blur_radius), 0)
                    # print(f"🎨 马赛克生成: 原图 {w}x{h} -> 小图 {w_small}x{h_small} -> 马赛克 {w}x{h}")

                    # 应用马赛克到检测区域
                    if np.any(mask > 0):
                        img_np[mask > 0] = mosaic_full[mask > 0]
                        # print(f"✅ 马赛克应用成功: {np.sum(mask > 0)} 个像素被替换")
                    else:
                        print(f"⚠️ 掩膜为空，跳过马赛克处理")

                    # 保存处理后的帧
                    output_frame = os.path.join(
                        processed_dir, f"frame_{i+1:06d}.png")
                    Image.fromarray(img_np).save(output_frame)
                    # print(f"💾 保存处理后的帧: {output_frame}")
                else:
                    # 没有检测到NSFW内容，直接复制原帧
                    output_frame = os.path.join(
                        processed_dir, f"frame_{i+1:06d}.png")
                    import shutil
                    shutil.copy2(frame_file, output_frame)

            except Exception as e:
                print(f"⚠️ 帧 {i+1} 处理出错: {e}")
                # 出错时复制原帧
                output_frame = os.path.join(
                    processed_dir, f"frame_{i+1:06d}.png")
                import shutil
                shutil.copy2(frame_file, output_frame)
                continue

        print(
            f"📊 检测统计: {nsfw_frames}/{total_frames} 帧包含NSFW内容 ({(nsfw_frames/total_frames)*100:.1f}%)")

        # 5. 合成视频
        print("🎬 合成处理后的视频...")
        processed_frames_pattern = os.path.join(
            processed_dir, "frame_%06d.png")

        if audio_stream:
            # 有音频：先合成视频，再合并音频
            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            cmd_video = [
                "ffmpeg", "-y", "-framerate", str(
                    fps), "-i", processed_frames_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", temp_video
            ]
            result = subprocess.run(cmd_video, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ 视频合成失败: {result.stderr}")
                return False, None

            # 合并音频
            cmd_audio = [
                "ffmpeg", "-y", "-i", temp_video, "-i", video_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-f", "mp4", output_path
            ]
            result = subprocess.run(cmd_audio, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ 音频合并失败: {result.stderr}")
                return False, None
        else:
            # 无音频：直接合成视频
            cmd_video = [
                "ffmpeg", "-y", "-framerate", str(
                    fps), "-i", processed_frames_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", output_path
            ]
            result = subprocess.run(cmd_video, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ 视频合成失败: {result.stderr}")
                return False, None

        print(f"✅ 视频处理完成!")
        print(f"📊 最终统计:")
        print(f"   - 总帧数: {total_frames}")
        print(f"   - NSFW帧数: {nsfw_frames}")
        print(f"   - NSFW比例: {(nsfw_frames/total_frames)*100:.1f}%")
        print(f"   - 输出视频: {output_path}")
        print(f"📁 检测结果保存在: {detection_dir}")

        return True, output_path

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return False, None

    finally:
        # 清理临时文件（保留检测结果）
        if os.path.exists(temp_dir):
            # 保留检测结果目录，只清理帧提取目录
            frames_dir = os.path.join(temp_dir, "frames")
            processed_dir = os.path.join(temp_dir, "processed")

            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)

            print("🧹 已清理临时帧文件")
            print(f"📁 检测结果保存在: {detection_dir}")


def nsfw_censor_video_nudenet_advanced(
    video_path: str,
    output_path: str = None,
    conf: float = 0.5,
    model_path: str = None,
    inference_resolution: int = 320,
    temp_dir: str = "temp_frames_nudenet",
    labels: list = None,
    mosaic_block: int = 8,
    mask_expand: float = 0
):
    """
    基于NudeNet的NSFW视频检测和打码功能, 支持过渡阶段掩膜传播马赛克
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径，默认为原文件名_censored.mp4
        conf: 置信度阈值
        model_path: NudeNet模型路径，None使用默认320n模型
        inference_resolution: 推理分辨率
        temp_dir: 临时帧目录
        labels: 要检测的标签列表，None使用所有标签
        mosaic_block: 马赛克块数量（数量越多越清晰）
        mask_expand: 掩膜扩张比例
    
    Returns:
        tuple: (是否成功, 输出视频路径)
    """
    try:
        from nudenet import NudeDetector
    except ImportError:
        print("❌ 请先安装NudeNet: pip install --upgrade 'nudenet>=3.4.2'")
        return False, None
    
    import subprocess
    import shutil
    import glob
    
    print(f"🎬 开始处理视频: {video_path}")
    
    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_censored_nudenet.mp4"
    
    # 确保输出路径有正确的扩展名
    if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        output_path += '.mp4'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    frames_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    detection_dir = os.path.join(temp_dir, "detections")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(detection_dir, exist_ok=True)
    
    try:
        # 1. 使用FFmpeg提取视频信息
        print("📹 获取视频信息...")
        cmd_info = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd_info, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 无法获取视频信息: {result.stderr}")
            return False, None
        
        import json
        video_info = json.loads(result.stdout)
        video_stream = None
        audio_stream = None
        
        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
            elif stream['codec_type'] == 'audio':
                audio_stream = stream
        
        if not video_stream:
            print("❌ 未找到视频流")
            return False, None
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # 尝试获取帧率，如果失败则使用默认30fps
        try:
            fps = eval(video_stream['r_frame_rate'])  # 计算帧率
            if fps <= 0 or fps > 60:  # 帧率异常时使用默认值
                fps = 30.0
                print(f"⚠️ 检测到异常帧率，使用默认30fps")
        except (ValueError, ZeroDivisionError, KeyError):
            fps = 30.0
            print(f"⚠️ 无法获取帧率，使用默认30fps")
        
        print(f"📹 视频信息: {width}x{height}, {fps:.2f}fps")
        if audio_stream:
            print(f"🎵 音频流: {audio_stream['codec_name']}")
        
        # 2. 提取所有帧
        print("🎬 提取视频帧...")
        cmd_extract = [
            "ffmpeg", "-i", video_path,  # 不添加fps滤镜，提取所有原始帧
            os.path.join(frames_dir, "frame_%06d.png")
        ]
        result = subprocess.run(cmd_extract, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 帧提取失败: {result.stderr}")
            return False, None
        
        # 获取帧文件列表
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        total_frames = len(frame_files)
        print(f"📊 提取了 {total_frames} 帧")
        
        if total_frames == 0:
            print("❌ 未提取到任何帧")
            return False, None
        
        # 3. 加载NudeNet模型
        print("🤖 加载NudeNet检测模型...")
        if model_path:
            detector = NudeDetector(model_path=model_path, inference_resolution=inference_resolution)
            print(f"📦 使用自定义模型: {model_path}")
        else:
            detector = NudeDetector()
            print("📦 使用默认320n模型")
        
        # 设置检测标签
        if labels is None:
            target_labels = all_labels
            print(f"🎯 使用所有标签进行检测: {len(target_labels)} 个类别")
        else:
            target_labels = labels
            print(f"🎯 使用指定标签进行检测: {target_labels}")
        
        # 4. 逐帧处理
        print("🔍 开始逐帧检测和处理...")
        nsfw_frames = 0
        
        # 时间窗口掩膜传播参数
        time_window = int(fps)*2  # 前后fps帧 (约1秒)
        mask_history = []  # 存储历史掩膜
        
        for i, frame_file in enumerate(frame_files):
            if (i + 1) % 30 == 0:  # 每30帧显示进度
                progress = ((i + 1) / total_frames) * 100
                print(f"📊 处理进度: {i+1}/{total_frames} ({progress:.1f}%)")
            
            # 使用NudeNet检测
            try:
                detections = detector.detect(frame_file)
                
                # 过滤低置信度和指定标签的检测
                filtered_detections = [
                    d for d in detections 
                    if d['score'] >= conf and d['class'] in target_labels
                ]
                
                # 保存检测结果
                detection_result = {
                    'frame': i + 1,
                    'detections': filtered_detections,
                    'count': len(filtered_detections),
                    'target_labels': target_labels
                }
                
                detection_file = os.path.join(detection_dir, f"frame_{i+1:06d}.json")
                with open(detection_file, 'w', encoding='utf-8') as f:
                    json.dump(detection_result, f, ensure_ascii=False, indent=2)
                
                # 打印检测结果
                if filtered_detections:
                    detection_classes = [d['class'] for d in filtered_detections]
                    # print(f"📊 帧 {i+1:06d}: 检测到 {len(filtered_detections)} 个区域: {detection_classes}")
                else:
                    print(f"📊 帧 {i+1:06d}: 未检测到指定标签的NSFW内容")
                
                # 读取原始帧
                img_pil = Image.open(frame_file)
                img_np = np.array(img_pil)
                h, w = img_np.shape[:2]
                
                # 创建当前帧掩膜
                current_mask = np.zeros((h, w), dtype=np.uint8)
                
                # 如果有检测到NSFW内容，创建掩膜
                if filtered_detections:
                    nsfw_frames += 1
                    
                    # 为每个检测区域创建掩膜
                    for detection in filtered_detections:
                        # NudeNet坐标格式: [x, y, width, height]
                        x, y, width, height = detection['box']
                        print(f"📊 帧 {i+1:06d} 检测区域: {detection['class']} 置信度: {detection['score']:.3f} 坐标: [x={x}, y={y}, w={width}, h={height}]")
                        
                        # 转换为边界坐标
                        x1, y1 = int(x), int(y)
                        x2, y2 = int(x + width), int(y + height)
                        
                        # 确保坐标在图像范围内
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # 确保坐标有效
                        if x2 > x1 and y2 > y1:
                            # 创建矩形掩膜
                            current_mask[y1:y2, x1:x2] = 255
                        else:
                            print(f"⚠️ 无效坐标: [{x1}, {y1}, {x2}, {y2}]")
                
                # 时间窗口掩膜传播
                print(f"🕒 时间窗口掩膜传播: 当前帧 {i+1}, 历史掩膜数量: {len(mask_history)}")
                
                # 获取时间窗口内的历史掩膜
                window_masks = []
                for j in range(max(0, i - time_window), i):
                    if j < len(mask_history) and mask_history[j] is not None:
                        window_masks.append(mask_history[j])
                        # print(f"📚 使用历史掩膜: 帧 {j+1}")
                
                # 合并当前掩膜和历史掩膜
                final_mask = current_mask.copy()
                for hist_mask in window_masks:
                    final_mask = cv2.bitwise_or(final_mask, hist_mask)
                
                # 区域扩张（在合并所有掩膜后统一扩张）
                if mask_expand > 0 and np.any(final_mask > 0):
                    k = int(max(h, w) * mask_expand / 10)
                    if k > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
                        print(f"🔧 掩膜扩张: 核大小 {k}x{k}")
                
                print(f"📊 掩膜统计: 总像素 {h*w}, 当前掩膜 {np.sum(current_mask > 0)}, 最终掩膜 {np.sum(final_mask > 0)}")
                
                # 如果有任何掩膜，进行马赛克处理
                if np.any(final_mask > 0):
                    # 生成马赛克 - 修改逻辑：mosaic_block表示马赛克块数量，数量越多越清晰
                    h_small = max(1, mosaic_block)
                    w_small = max(1, mosaic_block)
                    small = cv2.resize(img_np, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    print(f"🎨 马赛克生成: 原图 {w}x{h} -> 小图 {w_small}x{h_small} -> 马赛克 {w}x{h} (块数量: {mosaic_block}x{mosaic_block})")
                    
                    # 添加模糊处理，让轮廓更柔和
                    if mosaic_block <= 16:  # 当块数量较少时，添加模糊
                        blur_radius = max(3, min(15, 32 // mosaic_block))  # 根据块数量调整模糊半径
                        # 确保核大小是正奇数
                        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                        mosaic_full = cv2.GaussianBlur(mosaic_full, (blur_radius, blur_radius), 0)
                        print(f"🔍 应用模糊处理: 模糊半径 {blur_radius}x{blur_radius}")
                    
                    # 应用马赛克到检测区域
                    img_np[final_mask > 0] = mosaic_full[final_mask > 0]
                    print(f"✅ 马赛克应用成功: {np.sum(final_mask > 0)} 个像素被替换")
                    
                    # 保存处理后的帧
                    output_frame = os.path.join(processed_dir, f"frame_{i+1:06d}.png")
                    Image.fromarray(img_np).save(output_frame)
                    print(f"💾 保存处理后的帧: {output_frame}")
                else:
                    # 没有掩膜，直接复制原帧
                    output_frame = os.path.join(processed_dir, f"frame_{i+1:06d}.png")
                    import shutil
                    shutil.copy2(frame_file, output_frame)
                    print(f"📋 无掩膜，直接复制原帧: {output_frame}")
                
                # 保存当前掩膜到历史记录
                mask_history.append(current_mask.copy())
                
                # 保持历史记录大小在合理范围内
                if len(mask_history) > time_window * 2:
                    mask_history.pop(0)
                    
            except Exception as e:
                print(f"⚠️ 帧 {i+1} 处理出错: {e}")
                # 出错时复制原帧
                output_frame = os.path.join(processed_dir, f"frame_{i+1:06d}.png")
                import shutil
                shutil.copy2(frame_file, output_frame)
                continue
        
        print(f"📊 检测统计: {nsfw_frames}/{total_frames} 帧包含NSFW内容 ({(nsfw_frames/total_frames)*100:.1f}%)")
        
        # 5. 合成视频
        print("🎬 合成处理后的视频...")
        processed_frames_pattern = os.path.join(processed_dir, "frame_%06d.png")
        
        if audio_stream:
            # 有音频：先合成视频，再合并音频
            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            cmd_video = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", processed_frames_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", temp_video
            ]
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 视频合成失败: {result.stderr}")
                return False, None
            
            # 合并音频
            cmd_audio = [
                "ffmpeg", "-y", "-i", temp_video, "-i", video_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-f", "mp4", output_path
            ]
            result = subprocess.run(cmd_audio, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 音频合并失败: {result.stderr}")
                return False, None
        else:
            # 无音频：直接合成视频
            cmd_video = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", processed_frames_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", output_path
            ]
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 视频合成失败: {result.stderr}")
                return False, None
        
        print(f"✅ 视频处理完成!")
        print(f"📊 最终统计:")
        print(f"   - 总帧数: {total_frames}")
        print(f"   - NSFW帧数: {nsfw_frames}")
        print(f"   - NSFW比例: {(nsfw_frames/total_frames)*100:.1f}%")
        print(f"   - 输出视频: {output_path}")
        print(f"📁 检测结果保存在: {detection_dir}")
        
        return True, output_path
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return False, None
    
    finally:
        # 清理临时文件（保留检测结果）
        if os.path.exists(temp_dir):
            # 保留检测结果目录，只清理帧提取目录
            frames_dir = os.path.join(temp_dir, "frames")
            processed_dir = os.path.join(temp_dir, "processed")
            
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)
            
            print("🧹 已清理临时帧文件")
            print(f"📁 检测结果保存在: {detection_dir}")


def base():
    model = YOLO("nsfw-seg-vagina-x.pt")
    # model = YOLO("nsfw-seg-breast-x.pt")
    # model = YOLO("nsfw-seg-penis-x.pt")
    results = model.predict("1.jpg", imgsz=832, conf=0.1)
    results[0].show()
    """
    image 1/1 /root/autodl-tmp/nsfw/1.jpg: 832x576 1 item, 13.0ms
    Speed: 2.4ms preprocess, 13.0ms inference, 1.4ms postprocess per image at shape (1, 3, 832, 576)
    """



def nsfw_censor_video_ffmpeg(
    video_path: str,
    output_path: str = None,
    conf: float = 0.1,
    imgsz: int = 832,
    mask_expand: float = 0.5,
    mosaic_block: int = 8,
    temp_dir: str = "temp_frames"
):
    """
    基于FFmpeg的NSFW视频检测和打码功能
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径，默认为原文件名_censored.mp4
        conf: 置信度阈值
        imgsz: 模型输入尺寸
        mask_expand: 掩膜扩张比例
        mosaic_block: 马赛克块大小
        temp_dir: 临时帧目录
    
    Returns:
        tuple: (是否成功, 输出视频路径)
    """
    import subprocess
    import shutil
    import glob
    
    print(f"🎬 开始处理视频: {video_path}")
    
    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_censored.mp4"
    
    # 确保输出路径有正确的扩展名
    if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        output_path += '.mp4'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    frames_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # 1. 使用FFmpeg提取视频信息
        print("📹 获取视频信息...")
        cmd_info = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd_info, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 无法获取视频信息: {result.stderr}")
            return False, None
        
        import json
        video_info = json.loads(result.stdout)
        video_stream = None
        audio_stream = None
        
        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
            elif stream['codec_type'] == 'audio':
                audio_stream = stream
        
        if not video_stream:
            print("❌ 未找到视频流")
            return False, None
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # 尝试获取帧率，如果失败则使用默认30fps
        try:
            fps = eval(video_stream['r_frame_rate'])  # 计算帧率
            if fps <= 0 or fps > 60:  # 帧率异常时使用默认值
                fps = 30.0
                print(f"⚠️ 检测到异常帧率，使用默认30fps")
        except (ValueError, ZeroDivisionError, KeyError):
            fps = 30.0
            print(f"⚠️ 无法获取帧率，使用默认30fps")
        
        print(f"📹 视频信息: {width}x{height}, {fps:.2f}fps")
        if audio_stream:
            print(f"🎵 音频流: {audio_stream['codec_name']}")
        
        # 2. 提取所有帧
        print("🎬 提取视频帧...")
        cmd_extract = [
            "ffmpeg", "-i", video_path,  # 不添加fps滤镜，提取所有原始帧
            os.path.join(frames_dir, "frame_%06d.png")
        ]
        result = subprocess.run(cmd_extract, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 帧提取失败: {result.stderr}")
            return False, None
        
        # 获取帧文件列表
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        total_frames = len(frame_files)
        print(f"📊 提取了 {total_frames} 帧")
        
        if total_frames == 0:
            print("❌ 未提取到任何帧")
            return False, None
        
        # 3. 加载模型
        print("🤖 加载NSFW检测模型...")
        models = [
            YOLO("nsfw-seg-breast-x.pt"),
            YOLO("nsfw-seg-penis-x.pt"),
            YOLO("nsfw-seg-vagina-x.pt"),
        ]
        
        # 4. 逐帧处理
        print("🔍 开始逐帧检测和处理...")
        nsfw_frames = 0
        
        for i, frame_file in enumerate(frame_files):
            if (i + 1) % 30 == 0:  # 每30帧显示进度
                progress = ((i + 1) / total_frames) * 100
                print(f"📊 处理进度: {i+1}/{total_frames} ({progress:.1f}%)")
            
            # 初始化掩膜
            final_mask = np.zeros((height, width), np.uint8)
            detection_count = 0
            
            # 使用三个模型检测（直接使用文件路径）
            model_detections = [0, 0, 0]  # 记录每个模型的检测数量
            model_names = ["breast", "penis", "vagina"]  # 模型名称
            
            for j, model in enumerate(models):
                try:
                    results = model.predict(frame_file, imgsz=imgsz, conf=conf, verbose=False)
                    
                    # 保存每个模型的检测结果图片  前期测试
                    model_result_dir = os.path.join(temp_dir, f"model_{j+1}_{model_names[j]}")
                    os.makedirs(model_result_dir, exist_ok=True)
                    model_result_path = os.path.join(model_result_dir, f"frame_{i+1:06d}.jpg")
                    results[0].save(model_result_path)
                    #  测试完成
                    
                    if results[0].masks is not None and len(results[0].masks) > 0:
                        model_detections[j] = len(results[0].masks)
                        detection_count += len(results[0].masks)
                        
                        masks = results[0].masks.data.cpu().numpy()
                        for mask in masks:
                            # 确保mask是二值化的
                            mask = (mask > 0.5).astype(np.uint8)
                            # 调整mask尺寸到原图尺寸
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            
                            # 区域扩张
                            if mask_expand > 0:
                                k = int(max(height, width) * mask_expand / 10)
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                                mask = cv2.dilate(mask, kernel, iterations=1)
                            
                            # 合并到总掩膜
                            final_mask = cv2.bitwise_or(final_mask, mask)
                except Exception as e:
                    print(f"⚠️ 模型 {j+1} 检测出错: {e}")
                    continue
            
            # 打印每一帧的检测结果
            print(f"📊 帧 {i+1:06d}: 模型1={model_detections[0]}, 模型2={model_detections[1]}, 模型3={model_detections[2]}, 总计={detection_count}")
            
            # 如果有检测到NSFW内容，进行马赛克处理
            if detection_count > 0:
                nsfw_frames += 1
                
                # 读取原始帧进行马赛克处理
                img_pil = Image.open(frame_file)
                img_np = np.array(img_pil)
                
                # 生成马赛克 - 修改逻辑：mosaic_block表示马赛克块数量，数量越多越清晰
                h_small = max(1, mosaic_block)
                w_small = max(1, mosaic_block)
                small = cv2.resize(img_np, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                mosaic_full = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # 添加模糊处理，让轮廓更柔和
                if mosaic_block <= 16:  # 当块数量较少时，添加模糊
                    blur_radius = max(3, min(15, 32 // mosaic_block))  # 根据块数量调整模糊半径
                    # 确保核大小是正奇数
                    blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                    mosaic_full = cv2.GaussianBlur(mosaic_full, (blur_radius, blur_radius), 0)
                
                # 应用马赛克到检测区域
                img_np[final_mask == 1] = mosaic_full[final_mask == 1]
                
                # 保存处理后的帧
                output_frame = os.path.join(processed_dir, f"frame_{i+1:06d}.png")
                Image.fromarray(img_np).save(output_frame)
            else:
                # 没有检测到NSFW内容，直接复制原帧
                output_frame = os.path.join(processed_dir, f"frame_{i+1:06d}.png")
                import shutil
                shutil.copy2(frame_file, output_frame)
        
        print(f"📊 检测统计: {nsfw_frames}/{total_frames} 帧包含NSFW内容 ({(nsfw_frames/total_frames)*100:.1f}%)")
        
        # 5. 合成视频
        print("🎬 合成处理后的视频...")
        processed_frames_pattern = os.path.join(processed_dir, "frame_%06d.png")
        
        if audio_stream:
            # 有音频：先合成视频，再合并音频
            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            cmd_video = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", processed_frames_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", temp_video
            ]
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 视频合成失败: {result.stderr}")
                return False, None
            
            # 合并音频
            cmd_audio = [
                "ffmpeg", "-y", "-i", temp_video, "-i", video_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-f", "mp4", output_path
            ]
            result = subprocess.run(cmd_audio, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 音频合并失败: {result.stderr}")
                return False, None
        else:
            # 无音频：直接合成视频
            cmd_video = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", processed_frames_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", output_path
            ]
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 视频合成失败: {result.stderr}")
                return False, None
        
        print(f"✅ 视频处理完成!")
        print(f"📊 最终统计:")
        print(f"   - 总帧数: {total_frames}")
        print(f"   - NSFW帧数: {nsfw_frames}")
        print(f"   - NSFW比例: {(nsfw_frames/total_frames)*100:.1f}%")
        print(f"   - 输出视频: {output_path}")
        
        return True, output_path
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return False, None
    
    finally:
        # 清理临时文件（保留检测结果图片）
        if os.path.exists(temp_dir):
            # 保留检测结果目录，只清理帧提取目录
            frames_dir = os.path.join(temp_dir, "frames")
            processed_dir = os.path.join(temp_dir, "processed")
            
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)
            
            print("🧹 已清理临时帧文件")
            print(f"📁 检测结果保存在: {temp_dir}")
            print("   - model_1_breast/: 胸部检测结果")
            print("   - model_2_penis/: 阴茎检测结果") 
            print("   - model_3_vagina/: 阴道检测结果")




"""
# 使用示例

# 标准马赛克处理
nsfw_censor_image_nudenet(
    image_path="test.jpg",
    output_path="outputs/standard.jpg",
    conf=0.2,
    mosaic_block=8,
    mask_expand=0.2
)

# 日本AV风格马赛克处理
nsfw_censor_image_nudenet(
    image_path="test.jpg", 
    output_path="outputs/japanese_style.jpg",
    conf=0.2,
    mosaic_block=4,  # 较少的块数量，产生更强的马赛克效果
    mask_expand=0.2,
    japanese_style=True  # 启用日本风格处理
)

# 视频处理示例
nsfw_censor_video_nudenet(
    video_path="test_video.mp4",
    output_path="outputs/video_censored.mp4",
    conf=0.1,
    mosaic_block=8,
    mask_expand=0
)
"""