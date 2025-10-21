#!/usr/bin/env python3
"""
retinaface_video_extract.py

用法：
    python retinaface_video_extract.py --input path/to/input_dir --outdir path/to/output_dir

说明：
- 遍历输入文件夹中的所有视频（支持子目录递归）。
- 使用 RetinaFace 对每一帧进行人脸检测。
- 将检测到的人脸按固定大小裁剪并放在同一帧的网格中，保存为一个新视频，
  输出目录结构与输入目录结构一致。

依赖：
    pip install retinaface opencv-python-headless numpy
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from retinaface import RetinaFace
except Exception as e:
    print("错误：未能导入 retinaface 模块。请先安装：\n    pip install retinaface")
    sys.exit(1)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_faces_on_frame(detector, frame_rgb):
    try:
        res = detector.detect_faces(frame_rgb)
    except AttributeError:
        res = RetinaFace.detect_faces(frame_rgb)

    boxes = []
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(v, dict) and 'facial_area' in v:
                x1, y1, x2, y2 = map(int, v['facial_area'])
            elif all(k2 in v for k2 in ('x1','y1','x2','y2')):
                x1, y1, x2, y2 = int(v['x1']), int(v['y1']), int(v['x2']), int(v['y2'])
            else:
                continue
            boxes.append((x1, y1, x2, y2))
    elif isinstance(res, list):
        for r in res:
            if isinstance(r, (list, tuple)) and len(r) >= 4:
                x1, y1, x2, y2 = map(int, r[:4])
                boxes.append((x1, y1, x2, y2))
    return boxes

def frames_faces_to_grid(frames_faces, face_size, grid_cols):
    if len(frames_faces) == 0:
        return np.zeros((face_size, face_size * grid_cols, 3), dtype=np.uint8)

    resized = []
    for f in frames_faces[:grid_cols]:
        try:
            r = cv2.resize(f, (face_size, face_size))
        except Exception:
            r = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        resized.append(r)

    while len(resized) < grid_cols:
        resized.append(np.zeros((face_size, face_size, 3), dtype=np.uint8))

    row = np.hstack(resized)
    return row

def process_video(input_path: Path, output_path: Path, args):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print('无法打开视频：', input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grid_w = args.face_size * args.cols
    grid_h = args.face_size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))

    detector = RetinaFace

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if args.skip > 0 and (frame_idx - 1) % args.skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes = detect_faces_on_frame(detector, frame_rgb)
        except Exception as e:
            boxes = []

        faces_imgs = []
        for (x1, y1, x2, y2) in boxes:
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w - 1, x2)
            y2c = min(h - 1, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            face = frame[y1c:y2c, x1c:x2c]
            faces_imgs.append(face)

        grid_frame = frames_faces_to_grid(faces_imgs, args.face_size, args.cols)
        writer.write(grid_frame)
        written += 1

        if args.show_progress:
            print(f'视频 {input_path.name} - 帧 {frame_idx}, 检测到人脸: {len(faces_imgs)}', end='\r')

    cap.release()
    writer.release()

    print(f'\n完成：{output_path} ({written} 帧)')

def is_video_file(path: Path):
    return path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

def main():
    parser = argparse.ArgumentParser(description="递归遍历文件夹，对所有视频使用 RetinaFace 提取人脸并输出新视频。")
    parser.add_argument('--input', '-i', required=True, help='输入文件夹路径')
    parser.add_argument('--outdir', '-o', required=True, help='输出文件夹路径')
    parser.add_argument('--face-size', type=int, default=256)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--skip', type=int, default=0)
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
        out_path = output_dir / rel.parent / (v.stem + '_faces_grid.mp4')
        process_video(v, out_path, args)

if __name__ == '__main__':
    main()
