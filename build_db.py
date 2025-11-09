import os
# 强制设置镜像源 - 在导入任何其他库之前
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

# 然后才是其他import
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from tqdm import tqdm
import json

# 其余代码保持不变...
# src/build_db.py
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from tqdm import tqdm
import json


# 配置参数
class Config:
    video_dir = "../data/videos"  # 视频存放路径
    database_dir = "../data/database"  # 特征库输出路径
    model_name = "openai/clip-vit-base-patch32"  # 使用的CLIP模型
    num_frames = 8  # 每个视频采样的帧数
    device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_video_feature(video_path, model, processor, config):
    """
    从视频中提取CLIP语义特征向量
    """
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}. Skipping.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames. Skipping.")
        cap.release()
        return None

    # 2. 均匀采样帧
    frame_indices = np.linspace(0, total_frames - 1, config.num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            # 如果读取失败，用黑色图像填充
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()

    # 3. 使用CLIP处理图像
    inputs = processor(images=frames, return_tensors="pt", padding=True)

    # 4. 模型推理
    with torch.no_grad():
        inputs = {k: v.to(config.device) for k, v in inputs.items()}
        frame_features = model.get_image_features(**inputs)

    # 5. 平均池化得到视频特征
    video_feature = frame_features.mean(dim=0)
    return video_feature.cpu().numpy()


def main():
    config = Config()
    os.makedirs(config.database_dir, exist_ok=True)

    # 加载CLIP模型
    print(f"Loading CLIP model: {config.model_name} on {config.device}...")
    model = CLIPModel.from_pretrained(config.model_name).to(config.device)
    processor = CLIPProcessor.from_pretrained(config.model_name)
    model.eval()

    # 获取视频文件列表
    video_files = [f for f in os.listdir(config.video_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"Found {len(video_files)} videos.")

    video_features = {}
    failed_videos = []

    # 处理每个视频
    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(config.video_dir, video_file)
        feature = extract_video_feature(video_path, model, processor, config)

        if feature is not None:
            video_features[video_file] = feature.tolist()
        else:
            failed_videos.append(video_file)

    # 保存特征库
    feature_db_path = os.path.join(config.database_dir, "feature_database.json")
    with open(feature_db_path, 'w', encoding='utf-8') as f:
        json.dump(video_features, f, indent=4, ensure_ascii=False)

    # 保存视频列表
    video_list_path = os.path.join(config.database_dir, "video_list.json")
    with open(video_list_path, 'w', encoding='utf-8') as f:
        json.dump(list(video_features.keys()), f, indent=4, ensure_ascii=False)

    print(f"\nFeature database built successfully!")
    print(f"Saved to: {feature_db_path}")
    print(f"Processed {len(video_features)} videos. Failed: {len(failed_videos)}")
    if failed_videos:
        print("Failed videos:", failed_videos)


if __name__ == "__main__":
    main()