"""Detector-level runtime benchmark on a bundled sample video.

Measures pure inference latency (predict call, including ultralytics
pre/post-processing) per model on identical frames. This is NOT the full
pipeline benchmark (no tracking/OCR/violation logic) — it isolates the
detector so model choices can be compared on runtime as well as accuracy.

Usage:
    python scripts/benchmark_detectors.py --device cuda --frames 300
    python scripts/benchmark_detectors.py --device cpu --frames 100

Results are appended to benchmarks/detector_benchmark.csv together
with the hardware and software versions used, so numbers stay reproducible.
"""
import argparse
import csv
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO, RTDETR
from ultralytics.nn.modules import head, transformer

# The bundled RT-DETR checkpoint was trained with renamed Ultralytics
# classes (same alias hack as backend vehicle_detector.py).
head.RTDETRDecoderV2 = head.RTDETRDecoder
transformer.DeformableTransformerDecoderLayerV2 = transformer.DeformableTransformerDecoderLayer
transformer.MSDeformAttnV2 = transformer.MSDeformAttn

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CONFIG = REPO_ROOT / "backend" / "api" / "configs" / "pipeline.yml"
RESULTS_CSV = REPO_ROOT / "benchmarks" / "detector_benchmark.csv"

MODEL_CLASSES = {
    "yolo11": YOLO,
    "yolo26": YOLO,
    "rtdetrv2": RTDETR,
}


def load_frames(video_path, count, warmup):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    frames = []
    while len(frames) < count + warmup:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) <= warmup:
        raise SystemExit(f"Video too short: {len(frames)} frames")
    return frames


def benchmark_model(name, model_cfg, frames, warmup, device):
    model = MODEL_CLASSES[name](str(REPO_ROOT / "backend" / model_cfg["path"]))
    conf = model_cfg.get("conf_threshold", 0.5)
    iou = model_cfg.get("iou_threshold", 0.5)

    latencies_ms = []
    detections = 0
    for i, frame in enumerate(frames):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        results = model.predict(
            frame, verbose=False, device=device, conf=conf, iou=iou)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if i >= warmup:
            latencies_ms.append(elapsed_ms)
            detections += len(results[0].boxes)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "model": name,
        "frames": n,
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(latencies_ms[int(n * 0.95) - 1], 2),
        "fps": round(1000 / statistics.mean(latencies_ms), 1),
        "avg_detections": round(detections / n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="yolo11,yolo26",
                        help="comma-separated: yolo11,yolo26,rtdetrv2")
    parser.add_argument("--video", default=None,
                        help="video path (default: bundled camera 1 sample)")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    config = yaml.safe_load(PIPELINE_CONFIG.read_text())
    video = args.video or str(
        REPO_ROOT / "backend" / config["samples"]["1"]["video_path"])

    hardware = (torch.cuda.get_device_name(0)
                if args.device == "cuda" else platform.processor() or "cpu")
    print(f"device={args.device} ({hardware})")
    print(f"torch={torch.__version__} ultralytics={ultralytics.__version__}")
    print(f"video={video} frames={args.frames} warmup={args.warmup}\n")

    frames = load_frames(video, args.frames, args.warmup)

    rows = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in MODEL_CLASSES:
            raise SystemExit(f"Unsupported model for this benchmark: {name}")
        row = benchmark_model(name, config["models"][name],
                              frames, args.warmup, args.device)
        rows.append(row)
        print(f"{name:>10}: mean {row['mean_ms']}ms | median {row['median_ms']}ms "
              f"| p95 {row['p95_ms']}ms | {row['fps']} FPS "
              f"| {row['avg_detections']} det/frame")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "device", "hardware", "torch", "ultralytics",
                "video", "model", "frames", "mean_ms", "median_ms",
                "p95_ms", "fps", "avg_detections"])
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for row in rows:
            writer.writerow([
                stamp, args.device, hardware,
                torch.__version__, ultralytics.__version__,
                Path(video).name, row["model"], row["frames"],
                row["mean_ms"], row["median_ms"], row["p95_ms"],
                row["fps"], row["avg_detections"]])
    print(f"\nResults appended to {RESULTS_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
