import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from models.vision_cardio.physnet import PhysNet, hr_from_wave


def detect_face(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Expand to a square crop with some surrounding context.
    cx, cy = x + w / 2, y + h / 2
    side = int(max(w, h) * 1.3)
    x1, y1 = int(cx - side / 2), int(cy - side / 2)
    x2, y2 = x1 + side, y1 + side

    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--checkpoint",
        default="models/vision_cardio/rppg_physnet_ubfc.pt",
    )
    parser.add_argument("--output-dir", default="outputs/vision_cardio")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    clip_frames = int(ckpt["clip_frames"])
    clip_seconds = float(ckpt["clip_seconds"])
    sample_rate = clip_frames / clip_seconds

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration < clip_seconds:
        cap.release()
        raise SystemExit(
            f"Video is {duration:.2f}s; need at least "
            f"{clip_seconds:.2f}s."
        )

    # Use a contiguous 20-second window starting at the beginning.
    timestamps = np.arange(clip_frames) / sample_rate
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    crops = []
    last_face = None
    missed = 0

    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit(f"Could not read frame at {timestamp:.2f}s")

        face = detect_face(frame, detector)

        # Reuse the previous crop only for brief detection failures.
        # Fail rather than fabricate a crop at the start.
        if face is None:
            missed += 1
            if last_face is None:
                cap.release()
                raise SystemExit(
                    f"No face detected at {timestamp:.2f}s. "
                    "Try a clearer, front-facing video."
                )
            face = last_face
        else:
            last_face = face

        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        crops.append(face)

    cap.release()

    # (T,H,W,C) -> (B,C,T,H,W), RGB in [0,1].
    video = np.asarray(crops, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)

    model = PhysNet(width=int(ckpt["width"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    with torch.inference_mode():
        waveform = model(tensor).squeeze(0).cpu().numpy()

    if not np.isfinite(waveform).all():
        raise SystemExit("Model produced non-finite waveform values.")

    bpm = hr_from_wave(waveform, fs=sample_rate)
    if not np.isfinite(bpm):
        raise SystemExit("Could not estimate BPM from the waveform.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    waveform_path = out_dir / f"{stem}_waveform.csv"
    with waveform_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "rppg_value"])
        for t, value in zip(timestamps, waveform):
            writer.writerow([f"{t:.4f}", f"{value:.8f}"])

    print(f"Video: {video_path}")
    print(f"Window: {clip_seconds:.1f}s")
    print(f"Samples: {clip_frames} at {sample_rate:.2f} Hz")
    print(f"Face detections missed: {missed}/{clip_frames}")
    print(f"Estimated heart rate: {bpm:.1f} BPM")
    print(f"Waveform CSV: {waveform_path}")
    print("Device: CPU")
    print("Note: preliminary estimate; not a deepfake verdict.")


if __name__ == "__main__":
    main()
