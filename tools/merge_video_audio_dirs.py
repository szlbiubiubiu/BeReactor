#!/usr/bin/env python3
"""
Merge silent videos and audio files from two directory trees into a new output tree.

Assumption:
- video_dir and audio_dir have the same relative folder structure.
- each target pair shares the same stem (filename without extension) under the same folder.

Example:
  python tools/merge_video_audio_dirs.py \
    --video-dir /path/to/silent_videos \
    --audio-dir /path/to/audios \
    --output-dir /path/to/merged_videos
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch merge silent videos and audios with identical directory structure."
    )
    parser.add_argument("--video-dir", type=Path, required=True, help="Root directory of silent videos")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Root directory of audios")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory for merged videos")
    parser.add_argument(
        "--output-ext",
        type=str,
        default=".mp4",
        help="Output video extension (default: .mp4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations without running ffmpeg",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any video has no matched audio",
    )
    return parser.parse_args()


def ensure_ext(ext: str) -> str:
    if not ext.startswith("."):
        return f".{ext}"
    return ext


def collect_by_stem(root: Path, allowed_exts: Set[str]) -> Dict[Tuple[Path, str], Path]:
    mapping: Dict[Tuple[Path, str], Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in allowed_exts:
            continue
        rel = p.relative_to(root)
        key = (rel.parent, p.stem)
        if key in mapping:
            raise ValueError(
                f"Duplicate file stem under same folder: {key[0]}/{key[1]} (\n"
                f"  existing: {mapping[key]}\n  new: {p}\n)"
            )
        mapping[key] = p
    return mapping


def build_ffmpeg_cmd(video_path: Path, audio_path: Path, output_path: Path, overwrite: bool) -> List[str]:
    # Keep video stream as-is and encode audio to AAC for broad player compatibility.
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-v",
        "warning",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]


def merge_pairs(
    video_root: Path,
    audio_root: Path,
    output_root: Path,
    output_ext: str,
    overwrite: bool,
    dry_run: bool,
    strict: bool,
) -> int:
    video_map = collect_by_stem(video_root, VIDEO_EXTS)
    audio_map = collect_by_stem(audio_root, AUDIO_EXTS)

    total = len(video_map)
    merged = 0
    missing_audio = 0
    failed = 0

    keys: Iterable[Tuple[Path, str]] = sorted(video_map.keys(), key=lambda x: (str(x[0]), x[1]))

    for key in keys:
        rel_parent, stem = key
        video_path = video_map[key]
        audio_path = audio_map.get(key)

        if audio_path is None:
            missing_audio += 1
            print(f"[MISS] audio not found for: {video_path.relative_to(video_root)}")
            if strict:
                raise FileNotFoundError(
                    f"Audio not found for relative path: {rel_parent / (stem + video_path.suffix)}"
                )
            continue

        out_dir = output_root / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}{output_ext}"

        cmd = build_ffmpeg_cmd(video_path, audio_path, out_path, overwrite)
        if dry_run:
            print("[DRY]", " ".join(cmd))
            merged += 1
            continue

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            failed += 1
            print(f"[FAIL] {video_path.relative_to(video_root)}")
            if result.stderr.strip():
                print(result.stderr.strip())
            continue

        merged += 1
        print(f"[OK] {video_path.relative_to(video_root)} -> {out_path.relative_to(output_root)}")

    print("\n===== Summary =====")
    print(f"Videos found       : {total}")
    print(f"Merged successfully: {merged}")
    print(f"Missing audio      : {missing_audio}")
    print(f"Failed ffmpeg      : {failed}")

    return 0 if (failed == 0 and (missing_audio == 0 or not strict)) else 1


def main() -> int:
    args = parse_args()

    video_dir = args.video_dir.resolve()
    audio_dir = args.audio_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_ext = ensure_ext(args.output_ext.lower())

    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Invalid --video-dir: {video_dir}")
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise NotADirectoryError(f"Invalid --audio-dir: {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    return merge_pairs(
        video_root=video_dir,
        audio_root=audio_dir,
        output_root=output_dir,
        output_ext=output_ext,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        strict=args.strict,
    )


if __name__ == "__main__":
    raise SystemExit(main())
