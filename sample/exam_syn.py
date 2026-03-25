import os
import cv2
import torch
import librosa
import numpy as np
import ffmpeg
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch import nn

# -------------------------- 1. 精简版SyncNet模型（官方结构）--------------------------
class SyncNetLite(nn.Module):
    def __init__(self):
        super(SyncNetLite, self).__init__()
        # 视觉分支
        self.visual_net = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64, 128, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(128, 256, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten()
        )
        # 音频分支
        self.audio_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=(2,2), padding=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()
        )

    def forward(self, visual_feat, audio_feat):
        v_emb = self.visual_net(visual_feat)
        a_emb = self.audio_net(audio_feat)
        return v_emb, a_emb

# -------------------------- 2. 音视频预处理 --------------------------
def load_video(video_path, img_size=224, num_frames=20):
    try:
        # 提取音频
        audio, sr = librosa.load(
            ffmpeg.input(video_path).output('-', format='wav', ar=16000, ac=1).run(capture_stdout=True, quiet=True)[0],
            sr=16000
        )
        # 提取视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (img_size, img_size))
            frame = (frame / 255.0 - 0.5) * 2
            frames.append(frame.transpose(2, 0, 1))
        cap.release()
        if len(frames) == 0:
            return None, None
        # 格式对齐
        visual = torch.tensor(np.stack(frames), dtype=torch.float32).unsqueeze(0)
        audio_mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        audio_mel = torch.tensor(audio_mel[np.newaxis, np.newaxis, :, :20], dtype=torch.float32)
        return visual, audio_mel
    except:
        return None, None

# -------------------------- 3. SyncNet 指标计算 --------------------------
def calculate_sync_metrics(v_emb, a_emb):
    sync_score = torch.cosine_similarity(v_emb, a_emb).mean().item()
    l2_dist = torch.norm(v_emb - a_emb, p=2).item()
    pearson_r, _ = stats.pearsonr(v_emb.detach().numpy().flatten(), a_emb.detach().numpy().flatten())
    noise_emb = torch.randn_like(a_emb)
    y_true = [1, 0]
    y_score = [torch.cosine_similarity(v_emb, a_emb).item(), torch.cosine_similarity(v_emb, noise_emb).item()]
    auc = roc_auc_score(y_true, y_score)
    eer = round((1 - auc) * 100, 2)
    return {
        "Sync Score": round(sync_score, 4),
        "L2 Distance": round(l2_dist, 4),
        "Pearson": round(pearson_r, 4),
        "AUC": round(auc, 4),
        "EER (%)": eer
    }

# -------------------------- 4. 批量文件夹评测核心函数 --------------------------
def get_video_files(folder_path):
    """获取文件夹下所有视频文件"""
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(video_exts):
                video_files.append(os.path.join(root, f))
    return video_files

def batch_eval_folder(folder_path, weight_path="syncnet_v2.model"):
    # 加载模型
    model = SyncNetLite()
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
    model.eval()
    print(f"✅ 模型加载成功 | 开始批量评测文件夹：{folder_path}")

    # 获取所有视频
    video_list = get_video_files(folder_path)
    if len(video_list) == 0:
        print("❌ 未找到任何视频文件！")
        return
    print(f"✅ 共找到 {len(video_list)} 个视频\n")

    # 批量评测
    results = []
    for idx, video_path in enumerate(video_list, 1):
        video_name = os.path.basename(video_path)
        print(f"[{idx}/{len(video_list)}] 正在评测：{video_name}")

        # 预处理
        visual_feat, audio_feat = load_video(video_path)
        if visual_feat is None or audio_feat is None:
            print(f"❌ 视频读取失败，已跳过\n")
            results.append({"视频名称": video_name, "状态": "读取失败"})
            continue

        # 推理+计算指标
        try:
            with torch.no_grad():
                v_emb, a_emb = model(visual_feat, audio_feat)
            metrics = calculate_sync_metrics(v_emb, a_emb)
            metrics["视频名称"] = video_name
            metrics["状态"] = "评测成功"
            results.append(metrics)

            # 打印单视频结果
            print(f"✅ 评测完成：")
            for k, v in metrics.items():
                if k not in ["视频名称", "状态"]:
                    print(f"  {k}: {v}")
            print("-" * 60)
        except Exception as e:
            print(f"❌ 评测失败：{str(e)}\n")
            results.append({"视频名称": video_name, "状态": "评测失败"})

    # 保存汇总结果
    df = pd.DataFrame(results)
    df.to_csv("syncnet_batch_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n🎉 批量评测完成！所有结果已保存至：syncnet_batch_results.csv")

# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SyncNet 批量文件夹视频同步评测")
    parser.add_argument("--folder", required=True, help="视频文件夹路径")
    args = parser.parse_args()
    batch_eval_folder(args.folder)