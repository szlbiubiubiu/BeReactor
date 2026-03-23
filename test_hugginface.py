import os
# 第一步：强制设置HF镜像环境变量（必须在导入utmosv2前设置！）
# 优先级：代码内设置 > 系统环境变量，确保走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 开启多线程加速下载

# 第二步：升级依赖（可选但推荐，避免版本兼容问题）
# 先执行pip升级，再导入库
import subprocess
import sys
def upgrade_packages():
    try:
        # 升级huggingface_hub（utmosv2依赖它下载权重）
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub", "utmosv2"])
        print("依赖包升级完成")
    except Exception as e:
        print(f"依赖包升级失败（非必须，可忽略）：{e}")

# 执行升级（首次运行建议执行）
upgrade_packages()

# 第三步：导入并创建UTMOSv2模型
import utmosv2

try:
    # 创建预训练模型，此时会从hf-mirror.com下载权重
    model = utmosv2.create_model(pretrained=True)
    print("UTMOSv2预训练模型加载成功！")
except Exception as e:
    print(f"模型加载失败：{e}")
    # 备选方案：手动下载权重后加载（如果镜像仍无法下载）
    print("\n如果自动下载失败，可手动从https://hf-mirror.com下载权重，然后指定本地路径：")
    print("model = utmosv2.create_model(pretrained=False)")
    print("model.load_state_dict(torch.load('本地权重文件路径.pth'))")