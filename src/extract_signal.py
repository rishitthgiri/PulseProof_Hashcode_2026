"""
extract_signal.py  -  video  ->  per-region RGB signals (.npz)  ->  pulse report

Usage (run from the repo root):
    python src/extract_signal.py data/me.mp4                  # unknown label
    python src/extract_signal.py data/real_01.mp4 --label real
    python src/extract_signal.py data/fake_01.mp4 --label fake
    python src/extract_signal.py data/real_01_crf35.mp4 --cond crf35 --reuse-boxes real_01

Outputs
    cache/signals/{name}__{cond}.npz   <- the team contract (keys below)
    cache/rois/{name}.npz              <- face boxes, reusable for compressed copies
    cache/plots/{name}__{cond}.png     <- pulse traces + spectrum

.npz keys (the agreed contract):
    rgb_forehead, rgb_lcheek, rgb_rcheek : float32 (T, 3)   mean R,G,B per frame
    fps : float          label : int (real=0, fake=1, unknown=-1)
    detected_frac : float  (fraction of frames where a face was really found)
Note: "lcheek"/"rcheek" = left/right side of the IMAGE, not of the person.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import rppg  # noqa: E402

REGIONS = ["forehead", "lcheek", "rcheek"]
LABELS = {"real": 0, "fake": 1, "unknown": -1}

# MediaPipe face-mesh landmark groups that outline each region
MP_LANDMARKS = {
    "forehead": [10, 67, 69, 104, 108, 109, 151, 337, 338, 299, 297, 9],
    "lcheek":   [50, 101, 118, 117, 123, 187, 205, 36],
    "rcheek":   [280, 330, 347, 346, 352, 411, 425, 266],
}


# ------------------------------------------------------------ face detectors
class HaarDetector:
    """OpenCV Haar cascade. Works everywhere, but is jittery on tilted faces."""
    name = "haar"

    def __init__(self):
        path = ROOT / "haarcascade_frontalface_default.xml"
        if not path.exists():
            print("Downloading Haar cascade ...")
            url = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                   "data/haarcascades/haarcascade_frontalface_default.xml")
            urllib.request.urlretrieve(url, str(path))
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise RuntimeError("Could not load Haar cascade")

    def __call__(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])   # biggest face
        return {
            "forehead": (fx + .30 * fw, fy + .05 * fh, fx + .70 * fw, fy + .25 * fh),
            "lcheek":   (fx + .15 * fw, fy + .50 * fh, fx + .40 * fw, fy + .70 * fh),
            "rcheek":   (fx + .60 * fw, fy + .50 * fh, fx + .85 * fw, fy + .70 * fh),
        }


class MediaPipeDetector:
    """MediaPipe face mesh (legacy `solutions` API, needs mediapipe==0.10.14)."""
    name = "mediapipe"

    def __init__(self):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError("this mediapipe version has no legacy `solutions` API "
                               "(pip install mediapipe==0.10.14)")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])
        boxes = {}
        for name, idx in MP_LANDMARKS.items():
            sel = pts[idx]
            boxes[name] = (sel[:, 0].min(), sel[:, 1].min(), sel[:, 0].max(), sel[:, 1].max())
        return boxes


def make_detector(backend):
    if backend in ("auto", "mediapipe"):
        try:
            det = MediaPipeDetector()
            print("Face detector: MediaPipe")
            return det
        except Exception as e:                      # noqa: BLE001
            if backend == "mediapipe":
                raise
            print(f"MediaPipe unavailable ({e}); falling back to Haar.")
    det = HaarDetector()
    print("Face detector: Haar cascade")
    return det


# ------------------------------------------------------------- box handling
def fill_missing(boxes):
    """boxes: (T,3,4) with NaN where no face was found.
    Forward-fill, then back-fill the start, so we always have ONE box per frame
    (this keeps the signal evenly sampled, which the pulse maths needs)."""
    boxes = boxes.copy()
    T = len(boxes)
    valid = ~np.isnan(boxes[:, 0, 0])
    if not valid.any():
        return None
    first = np.argmax(valid)
    boxes[:first] = boxes[first]
    for t in range(first + 1, T):
        if not valid[t]:
            boxes[t] = boxes[t - 1]
    return boxes


def smooth_boxes(boxes, alpha=0.3):
    """Exponential moving average: stops the ROI jittering frame to frame."""
    out = boxes.copy()
    for t in range(1, len(out)):
        out[t] = alpha * boxes[t] + (1 - alpha) * out[t - 1]
    return out


def detect_boxes(video_path, detector, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    boxes = []
    while len(boxes) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        found = detector(frame)
        if found is None:
            boxes.append(np.full((3, 4), np.nan))
        else:
            boxes.append(np.array([found[r] for r in REGIONS], dtype=float))
    cap.release()
    return np.array(boxes)


def mean_rgb_from_boxes(video_path, boxes):
    cap = cv2.VideoCapture(str(video_path))
    T = len(boxes)
    sig = np.zeros((3, T, 3), dtype=np.float32)
    for t in range(T):
        ok, frame = cap.read()
        if not ok:                                  # video shorter than boxes
            sig[:, t] = sig[:, t - 1] if t else 0
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for r in range(3):
            x0, y0, x1, y1 = boxes[t, r]
            x0, x1 = int(max(0, round(x0))), int(min(w, round(x1)))
            y0, y1 = int(max(0, round(y0))), int(min(h, round(y1)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                sig[r, t] = sig[r, t - 1] if t else 0
            else:
                sig[r, t] = rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    cap.release()
    return sig


# --------------------------------------------------------------------- main
def process_video(video_path, label="unknown", cond="clean", backend="auto",
                  max_seconds=None, reuse_boxes=None, method="pos", show=False,
                  name=None, plot=True):
    video_path = Path(video_path)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    name = name or video_path.stem          # pass name= when many videos share a file name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps <= 1 or fps > 240:
        print("WARNING: fps metadata missing/odd, assuming 30.")
        fps = 30.0
    max_frames = int(max_seconds * fps) if max_seconds else 10**9
    print(f"{video_path.name}: ~{n_total} frames @ {fps:.2f} fps")

    cache = ROOT / "cache"
    for sub in ("signals", "rois", "plots"):
        (cache / sub).mkdir(parents=True, exist_ok=True)

    # ---- pass 1: face boxes (or reuse boxes from the clean video)
    if reuse_boxes:
        data = np.load(cache / "rois" / f"{reuse_boxes}.npz")
        boxes, det_frac = data["boxes"], float(data["detected_frac"])
        print(f"Reusing boxes from cache/rois/{reuse_boxes}.npz")
        if "frame_wh" in data.files:
            ow, oh = [float(v) for v in data["frame_wh"]]
            if (ow, oh) != (vid_w, vid_h):
                print(f"Video size {int(ow)}x{int(oh)} -> {vid_w}x{vid_h}: scaling boxes")
                boxes = boxes * np.array([vid_w / ow, vid_h / oh, vid_w / ow, vid_h / oh])
        else:
            print("WARNING: cached boxes have no frame size; if this video was resized, "
                  "re-run the clean video once to regenerate them.")
        if len(boxes) > max_frames:
            boxes = boxes[:max_frames]
    else:
        detector = make_detector(backend)
        raw = detect_boxes(video_path, detector, max_frames)
        det_frac = float(np.mean(~np.isnan(raw[:, 0, 0])))
        print(f"Face found in {det_frac * 100:.0f}% of {len(raw)} frames")
        if det_frac < 0.5:
            print("WARNING: face found in under half the frames. Use a clip with ONE person,\n"
                  "         face large and still, good light. Results will not be meaningful.")
        boxes = fill_missing(raw)
        if boxes is None:
            sys.exit("No face detected in any frame - cannot continue.")
        boxes = smooth_boxes(boxes)
        np.savez(cache / "rois" / f"{name}.npz", boxes=boxes, detected_frac=det_frac, fps=fps,
                 frame_wh=np.array([vid_w, vid_h], dtype=float))

    # ---- pass 2: mean RGB inside each box
    sig = mean_rgb_from_boxes(video_path, boxes)
    out = cache / "signals" / f"{name}__{cond}.npz"
    np.savez(out,
             rgb_forehead=sig[0], rgb_lcheek=sig[1], rgb_rcheek=sig[2],
             fps=np.float64(fps), label=np.int64(LABELS[label]),
             detected_frac=np.float64(det_frac))
    print(f"Saved signals -> {out.relative_to(ROOT)}  (T={sig.shape[1]})")

    # ---- pulse report
    if sig.shape[1] < int(5 * fps):
        print("Clip is shorter than 5 s: heart-rate estimate will be unreliable.")
    res = rppg.analyze_regions(
        {"forehead": sig[0], "lcheek": sig[1], "rcheek": sig[2]}, fps, method)
    print("\nRegion      bpm    SNR(dB)")
    for r in REGIONS:
        print(f"{r:<10} {res['bpm'][r]:6.1f}  {res['snr_db'][r]:7.1f}")
    print(f"bpm spread across regions: {res['bpm_spread']:.1f}")
    for k, v in res["coherence"].items():
        print(f"coherence {k}: {v:.2f}")
    if plot:
        png = cache / "plots" / f"{name}__{cond}.png"
        rppg.plot_report(res, fps, f"{name} [{cond}] ({method.upper()})", png, show)
        print(f"Saved plot    -> {png.relative_to(ROOT)}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract rPPG signals from a face video.")
    ap.add_argument("video")
    ap.add_argument("--label", choices=list(LABELS), default="unknown")
    ap.add_argument("--cond", default="clean", help="condition tag, e.g. clean, crf28")
    ap.add_argument("--backend", choices=["auto", "mediapipe", "haar"], default="auto")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--reuse-boxes", default=None,
                    help="name of a clean video whose cached boxes to reuse")
    ap.add_argument("--method", choices=["pos", "chrom"], default="pos")
    ap.add_argument("--show", action="store_true", help="open the plot window")
    ap.add_argument("--name", default=None, help="output name (default: video file name)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    process_video(a.video, a.label, a.cond, a.backend, a.max_seconds,
                  a.reuse_boxes, a.method, a.show, a.name, not a.no_plot)
