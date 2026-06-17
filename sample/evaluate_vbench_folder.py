"""
使用 VBench 对文件夹中（包含子文件夹）的视频进行批量质量评测，并统计每个维度的平均分。

安装依赖：
    pip install vbench
    # 如需评测 object_class / human_action / multiple_objects 等语义维度，还需要：
    pip install detectron2@git+https://github.com/facebookresearch/detectron2.git

VBench 支持 custom_input 模式的维度（无需标准 prompt 列表）：
    subject_consistency, background_consistency, motion_smoothness,
    dynamic_degree, aesthetic_quality, imaging_quality

需要标准 prompt 匹配的维度（需要 VBench_full_info.json）：
    object_class, multiple_objects, human_action, color,
    spatial_relationship, scene, temporal_style, appearance_style,
    overall_consistency, temporal_flickering

用法示例：
    # 只评 imaging_quality + aesthetic_quality（最快，无需额外依赖）
    python evaluate_vbench_folder.py --video_dir /path/to/videos

    # 评估所有 custom_input 维度
    python evaluate_vbench_folder.py --video_dir /path/to/videos --dimensions all_custom

    # 指定评测维度
    python evaluate_vbench_folder.py --video_dir /path/to/videos \\
        --dimensions subject_consistency background_consistency motion_smoothness aesthetic_quality imaging_quality

    # 保存结果到 CSV 和 JSON
    python evaluate_vbench_folder.py --video_dir /path/to/videos --output_csv results.csv --output_json results.json

    # 指定 VBench_full_info.json 路径（默认从 vbench 包自动查找）
    python evaluate_vbench_folder.py --video_dir /path/to/videos --info_json ./VBench_full_info.json
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

# ── VBench 支持 custom_input 模式的维度（不需要标准 prompt 列表）────────────
CUSTOM_INPUT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

# ── 所有 VBench 维度（含需要标准 prompt 的语义维度）─────────────────────────
ALL_DIMENSIONS = CUSTOM_INPUT_DIMENSIONS + [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
    "temporal_flickering",
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


def find_video_files(root_dir: str) -> list:
    """递归查找 root_dir 下所有视频文件。"""
    videos = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(dirpath, fname))
    return videos


def collect_videos_flat(root_dir: str, tmp_dir: str) -> str:
    """
    VBench 的 custom_input 模式要求所有视频在同一个文件夹中。
    如果 root_dir 有嵌套子文件夹，则把所有视频用符号链接（或复制）
    平铺到一个临时目录中返回。
    """
    video_files = find_video_files(root_dir)
    if not video_files:
        return None, []

    # 检测是否已经是平铺结构（所有文件都在 root_dir 的直接子层）
    all_flat = all(os.path.dirname(f) == root_dir for f in video_files)
    if all_flat:
        return root_dir, video_files

    # 有子文件夹 → 建立临时平铺目录（用符号链接避免占用额外磁盘）
    print(f"[INFO] 检测到子文件夹结构，正在将 {len(video_files)} 个视频平铺到临时目录…")
    flat_dir = os.path.join(tmp_dir, "flat_videos")
    os.makedirs(flat_dir, exist_ok=True)

    seen_names = {}
    for fpath in tqdm(video_files, desc="平铺视频文件", unit="file", ncols=80):
        fname = os.path.basename(fpath)
        if fname in seen_names:
            # 文件名冲突：在文件名前拼接父文件夹前缀
            parent = os.path.basename(os.path.dirname(fpath))
            fname = f"{parent}__{fname}"
        seen_names[fname] = fpath
        dst = os.path.join(flat_dir, fname)
        try:
            os.symlink(os.path.abspath(fpath), dst)
        except OSError:
            shutil.copy2(fpath, dst)  # 符号链接失败时改为复制

    return flat_dir, video_files


def find_vbench_info_json() -> str | None:
    """尝试从 vbench 包安装目录找到 VBench_full_info.json。"""
    try:
        import vbench as _vb
        pkg_dir = os.path.dirname(_vb.__file__)
        candidate = os.path.join(pkg_dir, "VBench_full_info.json")
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    # 当前运行目录
    local = os.path.join(os.getcwd(), "VBench_full_info.json")
    if os.path.isfile(local):
        return local
    return None


def run_vbench_evaluate(
    videos_path: str,
    dimensions: list,
    output_dir: str,
    info_json: str,
    device: torch.device,
    mode: str = "custom_input",
) -> dict:
    """
    调用 VBench Python API 对 videos_path 中的视频评测 dimensions。
    返回各维度原始结果 dict。
    """
    from vbench import VBench

    my_vbench = VBench(device, info_json, output_dir)

    results = {}
    dim_pbar = tqdm(
        dimensions,
        desc="VBench 维度评测",
        unit="dim",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for dim in dim_pbar:
        eval_mode = mode
        # 语义类维度必须用 standard_mode 或 vbench_category
        if dim not in CUSTOM_INPUT_DIMENSIONS:
            eval_mode = "vbench_standard"

        dim_pbar.set_postfix_str(dim, refresh=True)
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"[VBench] 正在评测维度: {dim}  (mode={eval_mode})")
        tqdm.write(f"{'='*60}")
        try:
            my_vbench.evaluate(
                videos_path=videos_path,
                name=f"eval_{dim}",
                dimension_list=[dim],
                mode=eval_mode,
            )
            # 读取 VBench 写出的结果 JSON
            result_file = os.path.join(output_dir, f"eval_{dim}_eval_results.json")
            if os.path.isfile(result_file):
                with open(result_file, "r") as f:
                    dim_result = json.load(f)
                results[dim] = dim_result
                # 提取维度分数（VBench 结果格式为 {dim: [score, detail]}）
                if isinstance(dim_result, dict) and dim in dim_result:
                    score_val = dim_result[dim]
                    if isinstance(score_val, (list, tuple)):
                        score_val = score_val[0]
                    tqdm.write(f"  ✓ {dim}: {float(score_val):.4f}")
                else:
                    tqdm.write(f"  ✓ {dim}: 结果已保存，请查看 {result_file}")
            else:
                tqdm.write(f"  [WARN] 未找到结果文件: {result_file}")
                results[dim] = None
        except Exception as e:
            tqdm.write(f"  [ERROR] 评测 {dim} 失败: {e}")
            results[dim] = None

    dim_pbar.close()
    return results


def extract_scores(results: dict) -> dict:
    """从 VBench 原始结果中提取每个维度的标量分数。"""
    scores = {}
    for dim, val in results.items():
        if val is None:
            scores[dim] = None
            continue
        try:
            if isinstance(val, dict) and dim in val:
                inner = val[dim]
                scores[dim] = float(inner[0]) if isinstance(inner, (list, tuple)) else float(inner)
            elif isinstance(val, (int, float)):
                scores[dim] = float(val)
            else:
                scores[dim] = None
        except Exception:
            scores[dim] = None
    return scores


def print_summary(scores: dict, video_count: int):
    print("\n" + "=" * 60)
    print(f"VBench 评测汇总  (共 {video_count} 个视频)")
    print("=" * 60)
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    for dim, score in scores.items():
        tag = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {dim:<30s}  {tag}")
    if valid_scores:
        avg = sum(valid_scores.values()) / len(valid_scores)
        print("-" * 60)
        print(f"  {'平均分 (有效维度)':<30s}  {avg:.4f}  [{len(valid_scores)}/{len(scores)} 维度有效]")
    print("=" * 60)


def save_csv(scores: dict, video_count: int, csv_path: str):
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "score"])
        for dim, score in scores.items():
            writer.writerow([dim, f"{score:.6f}" if score is not None else "N/A"])
        writer.writerow([])
        valid = [v for v in scores.values() if v is not None]
        avg = sum(valid) / len(valid) if valid else None
        writer.writerow(["AVERAGE", f"{avg:.6f}" if avg is not None else "N/A"])
        writer.writerow(["total_videos", video_count])
    print(f"[CSV] 结果已保存: {csv_path}")


def save_json(scores: dict, video_count: int, json_path: str):
    valid = [v for v in scores.values() if v is not None]
    output = {
        "total_videos": video_count,
        "dimensions": scores,
        "average_score": sum(valid) / len(valid) if valid else None,
        "valid_dimension_count": len(valid),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 结果已保存: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 VBench 对文件夹中的视频进行批量质量评测。"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="包含视频文件的根目录（支持子文件夹）。"
    )
    parser.add_argument(
        "--dimensions", nargs="+", default=["aesthetic_quality", "imaging_quality"],
        help=(
            "要评测的 VBench 维度，用空格分隔。"
            "传 'all_custom' 评测全部 custom_input 兼容维度，"
            "传 'all' 评测所有 16 个维度（部分维度需要 detectron2）。\n"
            f"custom_input 维度: {CUSTOM_INPUT_DIMENSIONS}\n"
            f"全部维度: {ALL_DIMENSIONS}"
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default="./vbench_results",
        help="VBench 输出目录（默认: ./vbench_results）。"
    )
    parser.add_argument(
        "--info_json", type=str, default=None,
        help="VBench_full_info.json 路径（不传则自动从 vbench 包目录查找）。"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="运行设备（默认: cuda:0）。"
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="可选：将结果保存为 CSV 文件。"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="可选：将结果保存为 JSON 文件。"
    )
    args = parser.parse_args()

    # ── 解析维度 ──────────────────────────────────────────────────────────
    if len(args.dimensions) == 1 and args.dimensions[0] == "all_custom":
        dimensions = CUSTOM_INPUT_DIMENSIONS
    elif len(args.dimensions) == 1 and args.dimensions[0] == "all":
        dimensions = ALL_DIMENSIONS
    else:
        dimensions = args.dimensions
        invalid = [d for d in dimensions if d not in ALL_DIMENSIONS]
        if invalid:
            print(f"[ERROR] 未知维度: {invalid}")
            print(f"支持的维度: {ALL_DIMENSIONS}")
            sys.exit(1)

    print(f"评测维度 ({len(dimensions)}): {dimensions}")

    # ── 检查输入目录 ───────────────────────────────────────────────────────
    if not os.path.isdir(args.video_dir):
        print(f"[ERROR] 视频目录不存在: {args.video_dir}")
        sys.exit(1)

    # ── 查找 VBench info JSON ──────────────────────────────────────────────
    info_json = args.info_json or find_vbench_info_json()
    if info_json is None:
        print("[WARN] 未找到 VBench_full_info.json，请通过 --info_json 指定路径。")
        print("       语义类维度将跳过。仅评测 custom_input 维度。")
        dimensions = [d for d in dimensions if d in CUSTOM_INPUT_DIMENSIONS]
        # 创建一个空的占位文件
        info_json = os.path.join(args.output_dir, "_dummy_info.json")
        os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.isfile(info_json):
            with open(info_json, "w") as f:
                json.dump({}, f)

    # ── 设备 ───────────────────────────────────────────────────────────────
    if "cuda" in args.device and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，切换到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── 平铺视频目录（VBench 要求同一文件夹）─────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="vbench_")
    try:
        flat_dir, video_files = collect_videos_flat(args.video_dir, tmp_dir)
        if not video_files:
            print(f"[ERROR] 在 {args.video_dir} 中未找到视频文件。")
            sys.exit(1)
        print(f"共找到 {len(video_files)} 个视频文件。")

        os.makedirs(args.output_dir, exist_ok=True)

        # ── 运行 VBench 评测 ───────────────────────────────────────────────
        raw_results = run_vbench_evaluate(
            videos_path=flat_dir,
            dimensions=dimensions,
            output_dir=args.output_dir,
            info_json=info_json,
            device=device,
            mode="custom_input",
        )

        # ── 汇总 ──────────────────────────────────────────────────────────
        scores = extract_scores(raw_results)
        print_summary(scores, len(video_files))

        if args.output_csv:
            save_csv(scores, len(video_files), args.output_csv)
        if args.output_json:
            save_json(scores, len(video_files), args.output_json)

    finally:
        # 清理临时平铺目录
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
