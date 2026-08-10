# TraffWise Rework Roadmap — Model & AI Pipeline

**Goal**: Upgrade the AI pipeline so the project works as both a portfolio piece
(every claim backed by a measured number) and a genuinely usable system (the
pipeline stays runnable after every phase).

**Strategy**: Incremental refactor. No big-bang rewrite. Each phase lands as a
self-contained, PR-sized change with before/after evidence. The scope is the
model/AI pipeline; backend and frontend architecture stay as-is unless an AI
change requires an adjustment.

**Local training hardware**: NVIDIA RTX 5060 (8 GB VRAM), 15 GB RAM — enough to
train nano/small detectors at 640 px locally.

**Training data**: `vehicle_detection.rar` (local only, gitignored) —
Roboflow-style YOLO export, 16,796 train / 4,213 val images at 640×640, classes
`bus=0, car=1, motorbike=2, truck=3` (matches
`backend/api/source/process/dataloader.py`). The archive has no `data.yaml`;
one must be generated before training.

---

## Phase 1 — Benchmark baseline

The README currently states that runtime latency and FPS are intentionally
omitted because they were never measured. Every later phase justifies itself
against this baseline, so it comes first.

- [ ] Build a benchmark harness that runs the real pipeline
      (`Controller`) over the bundled sample videos and records:
  - end-to-end FPS,
  - per-stage latency (detection / tracking / OCR / violation logic),
  - detection counts and track ID-switch statistics.
- [ ] Support model switching (YOLO11 / RT-DETRv2 / Faster R-CNN) so the
      existing accuracy table gains a runtime column.
- [ ] Record hardware and settings alongside every run (GPU, resolution,
      frame-skip config) so numbers are reproducible.
- [ ] Store results as CSV/JSON in the repo; add the baseline table to the
      README.

**Exit criteria**: one command produces a reproducible benchmark report for the
current pipeline on documented hardware.

## Phase 2 — Tracker upgrade (DeepSORT → ByteTrack / BoT-SORT)

`deep-sort-realtime` (2021) runs a separate appearance-embedding network per
frame and is the oldest link in the pipeline. Ultralytics ships ByteTrack and
BoT-SORT built in — typically faster and with fewer ID switches. Code-only
change, no training required.

- [ ] Abstract the tracker interface so DeepSORT and the new tracker are
      swappable behind one API.
- [ ] Integrate ByteTrack (and optionally BoT-SORT) via Ultralytics.
- [ ] Re-run the Phase 1 benchmark; compare FPS and ID-switch counts.
- [ ] Make the winner the default; keep DeepSORT selectable for comparison.

**Exit criteria**: measured tracker comparison table; default tracker chosen
from the numbers, not vibes.

## Phase 3 — License-plate OCR quality

Generic EasyOCR underperforms on Vietnamese plates and is the biggest
"actually usable" pain point.

- [ ] Add multi-frame plate voting: aggregate OCR reads across a track ID and
      keep the consensus string (biggest accuracy win, no training needed).
- [ ] Normalize output against Vietnamese plate formats
      (e.g. `59A-123.45`) with validation regexes; reject impossible reads.
- [ ] Evaluate PaddleOCR as a drop-in replacement for EasyOCR on a labeled
      sample of plate crops; adopt if measurably better.
- [ ] (Stretch) Fine-tune the recognition model on Vietnamese plate crops.

**Exit criteria**: plate-accuracy evaluation set exists; measured improvement
over the EasyOCR baseline is documented.

## Phase 4 — Detector retraining on the local dataset

Extend the repo's existing model-comparison story with newer detectors trained
on the 21k-image dataset.

- [ ] Extract the dataset and generate `data.yaml` (4 classes, train/val
      splits as packaged).
- [ ] Retrain YOLO11s locally as the refreshed baseline (the current 5.5 MB
      checkpoint is YOLO11n-class; document the trade-off).
- [ ] Train at least one newer architecture (e.g. current-generation
      Ultralytics YOLO) on the identical split for a fair comparison.
- [ ] Update the README accuracy table with the new rows; keep per-epoch CSV
      logs as training evidence, matching the existing convention.
- [ ] Ship the winning checkpoint through the Hugging Face asset manifest.

**Exit criteria**: refreshed accuracy table with reproducible logs; default
runtime model chosen from validation metrics.

## Phase 5 — Runtime optimization

Tune the serving path using Phase 1 instrumentation to find real bottlenecks.

- [ ] FP16 inference where the hardware supports it.
- [ ] Re-tune `AdaptiveFrameSkipper` thresholds against measured load.
- [ ] Evaluate batching / resolution trade-offs per camera.
- [ ] Final before/after benchmark table (Phase 1 vs. Phase 5) in the README.

**Exit criteria**: documented end-to-end FPS improvement over the Phase 1
baseline on the same hardware.

---

## Working rules

1. **Measure first, change second** — no upgrade merges without a before/after
   number from the Phase 1 harness.
2. **Always runnable** — `docker compose up` and the native quick start must
   work after every phase.
3. **Evidence over claims** — every README statement links to a log, CSV, or
   benchmark output, matching the project's existing convention.
