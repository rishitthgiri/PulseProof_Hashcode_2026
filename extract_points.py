"""
extract_points.py - ONE pass over every video -> a small .npz of per-frame numbers.
After this, nothing needs the videos again (features, classifier, eye analysis all read the .npz).

    python extract_points.py --root data/clips --max-seconds 15
    python extract_points.py --root data/clips --max-seconds 15 --csv       # also per-frame CSVs

Labels are guessed from the path: a name containing fake / _fs / deepfake -> fake (1),
real / original -> real (0), anything else -> -1. Check cache/points_index.csv and fix labels there.
Safe to stop and re-run: finished videos are skipped (use --force to redo).
Output per video: cache/signals/{name}__mediapipe.npz   (works with features.py)
"""
import argparse, csv, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REG = ("forehead", "lcheek", "rcheek")
MP = {"forehead": [10, 67, 69, 104, 108, 109, 151, 337, 338, 299, 297, 9],
      "lcheek":   [50, 101, 118, 117, 123, 187, 205, 36],
      "rcheek":   [280, 330, 347, 346, 352, 411, 425, 266]}
EYE_R, EYE_L = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
EYE_COLS = ["ear_r", "ear_l", "iris_rx", "iris_ry", "iris_lx", "iris_ly", "nose_x", "nose_y", "face_w"]
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def guess_label(rel):
    s = str(rel).lower()
    if any(k in s for k in ("fake", "_fs", "deepfake", "faceswap", "manip")):
        return 1
    if any(k in s for k in ("real", "original", "pristine")):
        return 0
    return -1


def ear(p, idx):
    a = p[idx]
    return (np.linalg.norm(a[1] - a[5]) + np.linalg.norm(a[2] - a[4])) / (2 * np.linalg.norm(a[0] - a[3]) + 1e-9)


def iris_off(p, centre, c0, c1):
    mid, w = (p[c0] + p[c1]) / 2, np.linalg.norm(p[c0] - p[c1]) + 1e-9
    return (p[centre] - mid) / w


def process(path, mesh, max_seconds, width):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if (not fps or fps <= 1 or fps > 240) else fps
    n_total, W0, H0 = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    limit = int(max_seconds * fps) if max_seconds else 10 ** 9
    rgb, eyes, det, box, centres = [], [], [], None, []
    while len(det) < limit:
        ok, f = cap.read()
        if not ok:
            break
        if width and f.shape[1] > width:
            f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
        h, w = f.shape[:2]
        im = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = mesh.process(im)
        found = bool(res.multi_face_landmarks)
        if found:
            p = np.array([[l.x * w, l.y * h] for l in res.multi_face_landmarks[0].landmark])
            new = np.array([[p[MP[r], 0].min(), p[MP[r], 1].min(), p[MP[r], 0].max(), p[MP[r], 1].max()] for r in REG])
            box = new if box is None else 0.3 * new + 0.7 * box          # smoothed, stops jitter
            fw = np.linalg.norm(p[234] - p[454])
            eyes.append([ear(p, EYE_R), ear(p, EYE_L), *iris_off(p, 468, 33, 133), *iris_off(p, 473, 362, 263),
                         p[1, 0] / w, p[1, 1] / h, fw / w])
        else:
            eyes.append([np.nan] * len(EYE_COLS))
        det.append(found)
        if box is None:
            rgb.append(np.full((3, 3), np.nan))
        else:
            row = []
            for r in range(3):
                x0, y0, x1, y1 = box[r]
                x0, x1, y0, y1 = int(max(0, x0)), int(min(w, x1)), int(max(0, y0)), int(min(h, y1))
                row.append(im[y0:y1, x0:x1].reshape(-1, 3).mean(0) if x1 - x0 > 1 and y1 - y0 > 1 else np.full(3, np.nan))
            rgb.append(np.array(row))
            centres.append(((box[0, 0] + box[0, 2]) / 2, (box[0, 1] + box[0, 3]) / 2, box[0, 2] - box[0, 0]))
    cap.release()
    det = np.array(det)
    if det.sum() == 0:
        return None
    rgb = np.array(rgb, dtype=float)                                      # (T, 3 regions, 3 colours)
    first = int(np.argmax(~np.isnan(rgb[:, 0, 0])))
    rgb[:first] = rgb[first]                                              # back-fill the start
    for t in range(1, len(rgb)):                                          # forward-fill gaps
        bad = np.isnan(rgb[t])
        rgb[t][bad] = rgb[t - 1][bad]
    run = best = 0
    for d in det:
        run = 0 if d else run + 1
        best = max(best, run)
    c = np.array(centres)
    motion = float(np.mean(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)) / max(c[:, 2].mean(), 1)) if len(c) > 1 else 0.0
    bpp = path.stat().st_size * 8 / max(n_total, 1) / max(W0 * H0, 1)    # bits per pixel per frame (compression level)
    return dict(rgb_forehead=rgb[:, 0].astype(np.float32), rgb_lcheek=rgb[:, 1].astype(np.float32),
                rgb_rcheek=rgb[:, 2].astype(np.float32), fps=np.float64(fps), detected_frac=np.float64(det.mean()),
                longest_gap_s=np.float64(best / fps), bpp=np.float64(bpp), motion_rel=np.float64(motion),
                eyes=np.array(eyes, dtype=np.float32), eye_cols=np.array(EYE_COLS), detected=det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/clips")
    ap.add_argument("--max-seconds", type=float, default=15)
    ap.add_argument("--width", type=int, default=640, help="shrink frames to this width (0 = native). Faster, and plenty for pulse")
    ap.add_argument("--csv", action="store_true", help="also write a per-frame CSV for each video")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    try:
        import mediapipe as mp
        mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        sys.exit(f"MediaPipe FaceMesh not available ({e}). Use the Python 3.12 venv with mediapipe==0.10.14")
    root = Path(a.root)
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    print(f"{len(vids)} videos under {root}")
    sig = ROOT / "cache" / "signals"; sig.mkdir(parents=True, exist_ok=True)
    index = []
    for i, v in enumerate(vids, 1):
        rel = v.relative_to(root)
        name = "_".join(rel.with_suffix("").parts)
        out = sig / f"{name}__mediapipe.npz"
        label = guess_label(rel)
        if out.exists() and not a.force:
            d = np.load(out); r = dict(fps=float(d["fps"]), T=len(d["rgb_forehead"]), det=float(d["detected_frac"]))
            print(f"[{i}/{len(vids)}] {name}: already done")
        else:
            data = process(v, mesh, a.max_seconds, a.width)
            if data is None:
                print(f"[{i}/{len(vids)}] {name}: SKIPPED (cannot open or no face found)")
                continue
            np.savez_compressed(out, label=np.int64(label), **data)
            r = dict(fps=float(data["fps"]), T=len(data["rgb_forehead"]), det=float(data["detected_frac"]))
            print(f"[{i}/{len(vids)}] {name}: {r['T']} frames, face in {r['det']:.0%}, label {label}")
            if a.csv:
                cdir = ROOT / "cache" / "points_csv"; cdir.mkdir(parents=True, exist_ok=True)
                cols = ["frame", "t_s"] + [f"{g}_{c}" for g in REG for c in "RGB"] + EYE_COLS
                with open(cdir / f"{name}.csv", "w", newline="") as fh:
                    w = csv.writer(fh); w.writerow(cols)
                    for t in range(r["T"]):
                        w.writerow([t, round(t / r["fps"], 3)] + [round(float(x), 3) for g in REG for x in data["rgb_" + g][t]]
                                   + [round(float(x), 4) for x in data["eyes"][t]])
        index.append(dict(name=name, label=label, group=name, path=str(v), fps=round(r["fps"], 2),
                          frames=r["T"], detected_frac=round(r["det"], 3)))
    if index:
        with open(ROOT / "cache" / "points_index.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
        labs = [x["label"] for x in index]
        print(f"\n{len(index)} videos: {labs.count(0)} real, {labs.count(1)} fake, {labs.count(-1)} unlabelled")
        print("Check/fix labels in cache/points_index.csv")


if __name__ == "__main__":
    main()