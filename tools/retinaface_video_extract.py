#!/usr/bin/env python3
"""
deepface_video_extract.py

用法：
    python deepface_video_extract.py --input path/to/input_dir --outdir path/to/output_dir

说明：
- 使用 Sefik Serengil 的 DeepFace（https://github.com/serengil/deepface）。
- 遍历输入文件夹中的所有视频（支持子目录递归）。
- 对每一帧使用 DeepFace 提取人脸。
- 将检测到的人脸按人脸编号分别保存为独立视频文件（使用 imageio.mimwrite）。
- 输出文件夹结构与输入保持一致。
- 使用 GPU（如果可用）。

依赖：
    pip install deepface opencv-python-headless numpy imageio torch torchvision
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import imageio
from deepface import DeepFace
# import torch

# # 设置 GPU 可用
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'使用设备: {device}')

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_faces_on_frame(frame_rgb):
    faces = []
    try:
        # DeepFace.extract_faces 返回 list of dict with 'face' key
        results = DeepFace.extract_faces(frame_rgb, detector_backend='mtcnn', enforce_detection=False, align=False)
        for res in results:
            if 'face' in res and res['face'] is not None:
                faces.append(res['face'])
    except Exception as e:
        print('检测错误:', e)
    return faces

def process_video(input_path: Path, output_dir: Path, args):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print('无法打开视频：', input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx = 0
    face_frames = {}

    ensure_dir(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if args.skip > 0 and (frame_idx - 1) % args.skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces_on_frame(frame_rgb)

        for i, face in enumerate(faces):
            if i not in face_frames:
                face_frames[i] = []
            face_frames[i].append(face)

        if args.show_progress:
            print(f'视频 {input_path.name} - 帧 {frame_idx}, 检测到人脸: {len(faces)}', end='\r')

    cap.release()

    for i, frames in face_frames.items():
        face_video_path = output_dir / f"{input_path.stem}_face_{i+1}.mp4"
        imageio.mimwrite(str(face_video_path), frames, fps=fps, macro_block_size=None)

    print(f'\n完成提取：{input_path} -> {output_dir}')

def is_video_file(path: Path):
    return path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

def main():
    parser = argparse.ArgumentParser(description="递归遍历文件夹，对所有视频使用 DeepFace 检测并保存人脸视频。")
    parser.add_argument('--input', '-i', required=True, help='输入文件夹路径')
    parser.add_argument('--outdir', '-o', required=True, help='输出文件夹路径')
    parser.add_argument('--skip', type=int, default=0, help='每隔多少帧处理一次')
    parser.add_argument('--show-progress', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.outdir)

    if not input_dir.exists():
        print('输入文件夹不存在:', input_dir)
        sys.exit(1)

    video_files = [p for p in input_dir.rglob('*') if is_video_file(p)]
    print(f'找到 {len(video_files)} 个视频文件。')

    for v in video_files:
        rel = v.relative_to(input_dir)
        out_path = output_dir / rel.parent
        process_video(v, out_path, args)

if __name__ == '__main__':
    main()
