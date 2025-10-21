#!/usr/bin/env python3
"""
retinaface_video_extract_insightface.py

用法：
    python retinaface_video_extract_insightface.py --input path/to/input_dir --outdir path/to/output_dir

说明：
- 使用 InsightFace (PyTorch + ONNX) 的 RetinaFace 实现。
- 自动使用 GPU（CUDAExecutionProvider）。
- 遍历输入文件夹，对所有视频进行人脸检测与裁剪。
- 每个人脸保存为独立的视频文件。
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print("错误：未能导入 insightface 模块。请先安装：\n    pip install insightface onnxruntime-gpu")
    sys.exit(1)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def init_face_detector():
    """初始化 InsightFace RetinaFace 检测器"""
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def detect_faces(app, frame):
    """返回 [(x1, y1, x2, y2), ...]"""
    faces = app.get(frame)
    boxes = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        boxes.append((x1, y1, x2, y2))
    return boxes

def process_video(app, input_path: Path, output_dir: Path, args):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print('无法打开视频：', input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    face_writers = {}
    ensure_dir(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if args.skip > 0 and (frame_idx - 1) % args.skip != 0:
            continue

        boxes = detect_faces(app, frame)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w - 1, x2), min(h - 1, y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            face = frame[y1c:y2c, x1c:x2c]

            # 初始化该人脸的视频写入器
            out_path = output_dir / f"{input_path.stem}_face_{i+1}.mp4"
            if i not in face_writers:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h_f, w_f, _ = face.shape
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_f, h_f))
                face_writers[i] = writer

            # 写入帧
            face_writers[i].write(face)

        if args.show_progress:
            print(f'视频 {input_path.name} - 帧 {frame_idx}, 检测到人脸: {len(boxes)}', end='\\r')

    cap.release()
    for writer in face_writers.values():
        writer.release()
    print(f'\\n完成提取：{input_path} -> {output_dir}')

def is_video_file(path: Path):
    return path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

def main():
    parser = argparse.ArgumentParser(description="使用 InsightFace (PyTorch + ONNX) RetinaFace 进行人脸检测并保存人脸视频。")
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

    app = init_face_detector()

    video_files = [p for p in input_dir.rglob('*') if is_video_file(p)]
    print(f'找到 {len(video_files)} 个视频文件。')

    for v in video_files:
        rel = v.relative_to(input_dir)
        out_path = output_dir / rel.parent
        process_video(app, v, out_path, args)

if __name__ == '__main__':
    main()
