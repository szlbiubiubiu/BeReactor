import utmosv2
import huggingface_hub
from functools import partial

# 核心：重写huggingface_hub的下载函数，强制指定国内镜像endpoint
def override_hf_download():
    # 保存原始下载函数
    original_hf_hub_download = huggingface_hub.hf_hub_download
    original_snapshot_download = huggingface_hub.snapshot_download
    
    # 重写函数，添加endpoint参数（固定为hf-mirror.com）
    def custom_hf_hub_download(*args, **kwargs):
        kwargs["endpoint"] = "https://hf-mirror.com"  # 强制走国内镜像
        return original_hf_hub_download(*args, **kwargs)
    
    def custom_snapshot_download(*args, **kwargs):
        kwargs["endpoint"] = "https://hf-mirror.com"  # 强制走国内镜像
        return original_snapshot_download(*args, **kwargs)
    
    # 替换原始函数
    huggingface_hub.hf_hub_download = custom_hf_hub_download
    huggingface_hub.snapshot_download = custom_snapshot_download
    print("已强制绑定hf-mirror.com镜像到旧版本huggingface_hub")

# 第一步：先重写下载函数
override_hf_download()

# 第二步：再加载UTMOSv2模型（此时下载会走镜像）
try:
    model = utmosv2.create_model(pretrained=True)
    print("UTMOSv2模型加载成功！")
except Exception as e:
    print(f"加载失败：{e}")