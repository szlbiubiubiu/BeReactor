#!/usr/bin/env python3
"""
retinaface_video_extract.py

用法：
    python retinaface_video_extract.py --input path/to/input_dir --outdir path/to/output_dir

说明：
- 使用 Sefik Serengil 的 RetinaFace（https://github.com/serengil/retinaface）。
- 遍历输入文件夹中的所有视频（支持子目录递归）。
- 对每一帧使用 RetinaFace 检测人脸。
- 将检测到的人脸裁剪并放入同一帧的网格，输出为新视频。
- 输出文件夹结构与输入保持一致。

依赖：
    pip install retinaface opencv-python-headless numpy
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
try:
    from retinaface import RetinaFace
except Exception as e:
    print("错误：未能导入 retinaface 模块。请先安装：\n    pip install retinaface")
    sys.exit(1)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_faces_on_frame(frame_rgb):
    """使用 Sefik Serengil 的 RetinaFace 检测人脸，返回 [(x1, y1, x2, y2), ...]"""
    boxes = []
    try:
        results = RetinaFace.detect_faces(frame_rgb)
    except Exception as e:
        print("检测错误:", e)
        return boxes

    if isinstance(results, dict):
        for face_id, face_data in results.items():
            if 'facial_area' in face_data:
                x1, y1, x2, y2 = map(int, face_data['facial_area'])
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
        boxes = detect_faces_on_frame(frame_rgb)

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
    parser = argparse.ArgumentParser(description="递归遍历文件夹，对所有视频使用 RetinaFace (Sefik Serengil) 提取人脸并输出新视频。")
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

    for v in tqdm(video_files):
        rel = v.relative_to(input_dir)
        out_path = output_dir / rel.parent / (v.stem + '_faces_grid.mp4')
        process_video(v, out_path, args)

if __name__ == '__main__':
    main()