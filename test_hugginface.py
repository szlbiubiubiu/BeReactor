# 第一步：在所有导入前，优先设置HF镜像环境变量（0.24版本完美支持）
import os
# 强制指定国内镜像（0.24版本会优先读取这个变量）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 开启多线程下载（加速权重获取）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 第二步：验证环境变量是否生效（可选，用于排查）
print(f"HF_ENDPOINT = {os.getenv('HF_ENDPOINT')}")  # 应输出 https://hf-mirror.com
print(f"huggingface_hub版本 = {__import__('huggingface_hub').__version__}")  # 应输出 0.24.x

# 第三步：导入并加载UTMOSv2模型
import utmosv2

try:
    # 此时pretrained=True会从hf-mirror.com下载权重
    model = utmosv2.create_model(pretrained=True)
    print("\n✅ UTMOSv2模型加载成功！已通过hf-mirror.com下载权重")
except Exception as e:
    print(f"\n❌ 模型加载失败：{e}")
    # 兜底：手动指定镜像下载权重（0.24版本支持endpoint参数）
    from huggingface_hub import snapshot_download
    # 手动下载UTMOSv2权重到本地（指定镜像）
    local_repo = snapshot_download(
        repo_id="nii-yamagishilab/UTMOSv2",  # UTMOSv2的HuggingFace仓库地址
        endpoint="https://hf-mirror.com"     # 显式指定镜像（0.24版本支持）
    )
    # 从本地加载模型
    model = utmosv2.create_model(pretrained=False)
    model.load_state_dict(__import__('torch').load(f"{local_repo}/pytorch_model.bin"))
    print("✅ 已手动从镜像下载权重，并本地加载成功！")