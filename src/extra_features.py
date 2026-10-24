"""
extra_features.py - stitching-seam features + skin tone (ITA) per video.

    python src/extra_features.py --labels data/labels_hf.csv
    python src/extra_features.py --labels data/labels_hf.csv --root <folder with the HF videos>

Reads the ORIGINAL videos (not the .npz). Writes cache/extra_features.csv
Stitching idea: a face swap leaves a seam at the face outline. We compare three zones:
  inner band (just inside the oval), outer band (just outside), interior (rest of face)
on sharpness, noise and colour, plus how much those jump from frame to frame (flicker).
Skin tone: ITA angle from two cheek patches (a proxy for tone, NOT a Fitzpatrick label).
"""
import argparse, csv, re
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
FM = mp.solutions.face_mesh
OVAL = sorted({i for pair in FM.FACEMESH_FACE_OVAL for i in pair})
CHEEKS = (205, 425)


def lab_img(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def frame_feats(bgr, pts):
    h, w = bgr.shape[:2]
    hull = cv2.convexHull(pts[OVAL].astype(np.int32))
    m = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(m, hull, 1)
    fw = max(1, pts[OVAL, 0].max() - pts[OVAL, 0].min())
    k = max(3, int(fw * 0.04))
    el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    er, di = cv2.erode(m, el(k)), cv2.dilate(m, el(k))
    inner, outer = (m > 0) & (er == 0), (di > 0) & (m == 0)
    interior = cv2.erode(m, el(2 * k)) > 0
    if min(inner.sum(), outer.sum(), interior.sum()) < 50:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    res = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    lab = lab_img(bgr)
    sh = {n: float(lap[z].var()) + 1.0 for n, z in (("in", inner), ("out", outer), ("int", interior))}
    nz = {n: float(res[z].std()) + 1e-3 for n, z in (("in", inner), ("out", outer), ("int", interior))}
    mu = {n: lab[z].reshape(-1, 3).mean(0) for n, z in (("in", inner), ("out", outer), ("int", interior))}
    f = dict(
        sharp_in_int=np.log(sh["in"] / sh["int"]), sharp_in_out=np.log(sh["in"] / sh["out"]),
        noise_in_int=np.log(nz["in"] / nz["int"]), noise_int_out=np.log(nz["int"] / nz["out"]),
        col_dL_in_out=abs(mu["in"][0] - mu["out"][0]),
        col_dab_in_out=float(np.linalg.norm(mu["in"][1:] - mu["out"][1:])),
        col_dab_in_int=float(np.linalg.norm(mu["in"][1:] - mu["int"][1:])))
    # skin tone from cheek patches
    r = max(3, int(fw * 0.06)); Ls, bs = [], []
    for c in CHEEKS:
        x, y = pts[c].astype(int)
        patch = lab[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
        if len(patch):
            Ls.append(patch[:, 0].mean()); bs.append(patch[:, 2].mean())
    f["_L"], f["_b"] = float(np.mean(Ls)), float(np.mean(bs))
    return f


def video_feats(path, stride=2, max_frames=120):
    cap = cv2.VideoCapture(str(path)); fm = FM.FaceMesh(static_image_mode=False, max_num_faces=1)
    rows, n, seen = [], 0, 0
    while len(rows) < max_frames:
        ok, bgr = cap.read()
        if not ok: break
        n += 1
        if n % stride: continue
        seen += 1
        if bgr.shape[1] > 640:
            s = 640 / bgr.shape[1]; bgr = cv2.resize(bgr, None, fx=s, fy=s)
        r = fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks: continue
        h, w = bgr.shape[:2]
        pts = np.array([[l.x * w, l.y * h] for l in r.multi_face_landmarks[0].landmark])
        f = frame_feats(bgr, pts)
        if f: rows.append(f)
    cap.release(); fm.close()
    if len(rows) < 5: return None
    keys = [k for k in rows[0] if not k.startswith("_")]
    out = {f"st_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    for k in ("sharp_in_int", "col_dab_in_out"):                       # flicker over time
        v = np.array([r[k] for r in rows]); out[f"st_flick_{k}"] = float(v.std())
        out[f"st_jump_{k}"] = float(np.mean(np.abs(np.diff(v))))
    L, b = np.median([r["_L"] for r in rows]), np.median([r["_b"] for r in rows])
    out["ita"] = float(np.degrees(np.arctan2(L - 50.0, b)))
    out["st_face_frac"] = len(rows) / max(1, seen)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "data" / "labels_hf.csv"))
    ap.add_argument("--root", default=None, help="folder with HF videos (default: HF cache)")
    ap.add_argument("--out", default=str(ROOT / "cache" / "extra_features.csv"))
    a = ap.parse_args()
    if a.root: root = Path(a.root)
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download("NetherlandsForensicInstitute/rppg-deepfake-detection", repo_type="dataset"))
    vids = {"_".join(p.relative_to(root).with_suffix("").parts): p
            for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov")}
    names = [r["name"] for r in csv.DictReader(open(a.labels))]
    rows = []
    for i, n in enumerate(names, 1):
        if n not in vids: print(f"[{i}/{len(names)}] {n}: video not found, skipped"); continue
        f = video_feats(vids[n])
        print(f"[{i}/{len(names)}] {n}: {'ok' if f else 'too few face frames'}")
        if f: rows.append(dict(name=n, **f))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "name", k))
    w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, len(rows), "videos")
