"""
Evaluate audio files in a directory (including subdirectories) using UTMOSv2.

Usage:
    python evaluate_utmos_folder.py --audio_dir /path/to/audio_folder
    python evaluate_utmos_folder.py --audio_dir /path/to/audio_folder --model_path ./fold0_s42_best_model.pth
    python evaluate_utmos_folder.py --audio_dir /path/to/audio_folder --device cuda:1
    python evaluate_utmos_folder.py --audio_dir /path/to/audio_folder --output_csv results.csv
"""

import os
import argparse
import glob

import torch
import librosa
import utmosv2


AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}


def find_audio_files(root_dir: str) -> list:
    """Recursively find all audio files under root_dir."""
    audio_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(dirpath, fname))
    return audio_files


def load_utmos_model(model_path: str, device: torch.device):
    """Load and return the UTMOSv2 model."""
    model = utmosv2.create_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"UTMOSv2 model loaded from: {model_path}  (device={device})")
    return model


def evaluate_single(model, audio_path: str, sr: int = 16000) -> float | None:
    """Return the MOS score for a single audio file, or None on failure."""
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        mos_score = model.predict(data=audio, sr=sr)
        score = float(mos_score.item()) if hasattr(mos_score, 'item') else float(mos_score)
        return score
    except Exception as e:
        print(f"  [ERROR] Failed to evaluate {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute UTMOSv2 MOS scores for all audio files in a directory."
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True,
        help="Root directory containing audio files (subdirectories are scanned recursively)."
    )
    parser.add_argument(
        "--model_path", type=str, default="./fold0_s42_best_model.pth",
        help="Path to the UTMOSv2 pretrained weights (default: ./fold0_s42_best_model.pth)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run the model on (default: cuda:0). Use 'cpu' if no GPU is available."
    )
    parser.add_argument(
        "--sr", type=int, default=16000,
        help="Sampling rate for loading audio (default: 16000)."
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Optional path to save per-file results as a CSV."
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not os.path.isdir(args.audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")

    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')

    # ── Find audio files ─────────────────────────────────────────────────────
    audio_files = find_audio_files(args.audio_dir)
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return

    print(f"Found {len(audio_files)} audio file(s) under: {args.audio_dir}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_utmos_model(args.model_path, device)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results = []  # list of (path, score)
    total_score = 0.0
    valid_count = 0

    for idx, fpath in enumerate(audio_files, start=1):
        rel_path = os.path.relpath(fpath, args.audio_dir)
        score = evaluate_single(model, fpath, sr=args.sr)

        if score is not None:
            total_score += score
            valid_count += 1
            print(f"  [{idx}/{len(audio_files)}] {rel_path}  ->  MOS = {score:.4f}")
        else:
            print(f"  [{idx}/{len(audio_files)}] {rel_path}  ->  MOS = N/A")

        results.append((fpath, rel_path, score))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Total audio files:   {len(audio_files)}")
    print(f"Successfully scored: {valid_count}")
    print(f"Failed:              {len(audio_files) - valid_count}")

    if valid_count > 0:
        avg_mos = total_score / valid_count
        print(f"Average MOS Score:   {avg_mos:.4f}")
    else:
        avg_mos = None
        print("Average MOS Score:   N/A (no valid scores)")
    print("=" * 60)

    # ── Optional CSV output ──────────────────────────────────────────────────
    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'relative_path', 'mos_score'])
            for fpath, rel_path, score in results:
                writer.writerow([fpath, rel_path, score if score is not None else 'N/A'])
            # Write summary row
            writer.writerow([])
            writer.writerow(['SUMMARY', f'total={len(audio_files)}', f'valid={valid_count}',
                             f'avg_mos={avg_mos:.4f}' if avg_mos is not None else 'avg_mos=N/A'])
        print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
