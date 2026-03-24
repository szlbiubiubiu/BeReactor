import os
# 强制设置国内镜像（0.24版本完美支持）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 导入并加载正确仓库ID的UTMOSv2模型
import utmosv2

try:
    # 关键：使用正确的仓库ID对应的模型加载逻辑
    # utmosv2.create_model 内部已适配 sarulab-speech/UTMOSv2
    model = utmosv2.create_model(pretrained=True)
    print("✅ UTMOSv2模型加载成功！已从hf-mirror.com下载正确权重")
except Exception as e:
    print(f"❌ 加载失败：{e}")
    # 兜底：手动指定正确仓库ID+镜像下载
    from huggingface_hub import snapshot_download
    # 手动下载正确仓库的权重
    local_repo = snapshot_download(
        repo_id="sarulab-speech/UTMOSv2",  # 修正为正确仓库ID
        endpoint="https://hf-mirror.com"
    )
    # 本地加载
    (1)import torch
    model = utmosv2.create_model(pretrained=False)
    model.load_state_dict(torch.load('./fold0_s42_best_model.pth'))
    mos = model.predict(data=data, sr=16000) 

    (2)python syncnet.py --video [你的视频路径] --visualize
    (3)test ASR to text? better ASR would result in better results
    # model.load_state_dict(torch.load(f"{local_repo}/pytorch_model.bin"))
    
    print("✅ 已手动从镜像下载正确权重并加载成功！")