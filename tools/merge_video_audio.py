"""
递归扫描两个文件夹中的 .mp4 和 .wav 文件，按文件名（去除扩展名后）配对，
用 ffmpeg 将视频与音频合并，输出到指定的扁平目录（无子文件夹）。

用法:
    python merge_video_audio.py \
        --video_dir /path/to/mp4_folder \
        --audio_dir /path/to/wav_folder \
        --output_dir /path/to/output

可选参数:
    --suffix        输出文件名后缀，默认 "_merged"
    --overwrite     覆盖已存在的输出文件
"""

import argparse
import subprocess
import sys
from pathlib import Path


def collect_files(directory: Path, suffix: str) -> dict[str, Path]:
    """递归收集目录下所有指定后缀的文件，返回 {stem: path} 映射。"""
    return {p.stem: p for p in directory.rglob(f"*{suffix}")}


def merge(video_path: Path, audio_path: Path, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        print(f"[SKIP] 已存在: {output_path.name}")
        return True

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {video_path.name} + {audio_path.name}\n{result.stderr[-500:]}")
        return False
    print(f"[OK] {output_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="批量合并视频(.mp4)与音频(.wav)")
    parser.add_argument("--video_dir", required=True, help=".mp4 文件所在目录（递归扫描）")
    parser.add_argument("--audio_dir", required=True, help=".wav 文件所在目录（递归扫描）")
    parser.add_argument("--output_dir", required=True, help="输出目录（扁平，无子文件夹）")
    parser.add_argument("--suffix", default="_merged", help="输出文件名后缀，默认 '_merged'")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    if not video_dir.is_dir():
        sys.exit(f"[ERROR] video_dir 不存在: {video_dir}")
    if not audio_dir.is_dir():
        sys.exit(f"[ERROR] audio_dir 不存在: {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_files(video_dir, ".mp4")
    audios = collect_files(audio_dir, ".wav")

    matched = sorted(set(videos) & set(audios))
    only_video = sorted(set(videos) - set(audios))
    only_audio = sorted(set(audios) - set(videos))

    print(f"找到视频: {len(videos)} 个，音频: {len(audios)} 个，匹配: {len(matched)} 对")
    if only_video:
        print(f"[WARN] 仅有视频（无对应音频）: {only_video}")
    if only_audio:
        print(f"[WARN] 仅有音频（无对应视频）: {only_audio}")

    if not matched:
        sys.exit("[ERROR] 没有找到任何匹配的文件对，请检查文件名是否一致。")

    ok, fail = 0, 0
    for stem in matched:
        out_name = f"{stem}{args.suffix}.mp4"
        success = merge(videos[stem], audios[stem], output_dir / out_name, args.overwrite)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n完成: 成功 {ok} 个，失败 {fail} 个。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
