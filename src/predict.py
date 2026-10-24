"""
predict.py - video -> Real / Fake / Unreliable (+ reason)

    python src/predict.py data/clip.mp4 --backend mediapipe

Design: the gate asks "could a pulse even be measured in this video?" using quality
indicators that do NOT depend on the pulse itself (compression, face visibility, gaps,
length). The classifier then asks "does the pulse that WAS measured look real?".
If the gate used pulse strength, every fake with no pulse would be called Unreliable.
The thresholds below are starting values; the compression benchmark is what should set them.
"""
import argparse, sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import features as feat  # noqa: E402

MODEL_PATH = ROOT / "models" / "mvp.joblib"
DEFAULTS = dict(min_bpp=0.02, min_detect=0.8, max_gap_s=2.0, min_dur=5.0, min_conf=0.6)


def load_model(path=MODEL_PATH):
    if Path(path).exists():
        import joblib
        return joblib.load(path)
    return None


def decide(f, bundle=None, **thr):
    t = {**DEFAULTS, **thr}
    issues = []
    if f["duration_s"] < t["min_dur"]:
        issues.append(f"clip too short ({f['duration_s']:.1f} s)")
    if f["detected_frac"] < t["min_detect"]:
        issues.append(f"face found in only {f['detected_frac']:.0%} of frames")
    if f["longest_gap_s"] > t["max_gap_s"]:
        issues.append(f"face lost for {f['longest_gap_s']:.1f} s in a row")
    bpp = f.get("bpp", float("nan"))
    if bpp == bpp and bpp < t["min_bpp"]:
        issues.append(f"video too compressed to recover a pulse ({bpp:.3f} bits/pixel < {t['min_bpp']})")
    if issues:
        return dict(verdict="Unreliable", reason="; ".join(issues), p_fake=None)
    if bundle is None:
        return dict(verdict="Untrained", p_fake=None,
                    reason="video quality is fine, but no model is trained yet (needs labelled real and fake videos)")
    x = np.array([[f[c] for c in bundle["features"]]])
    p = float(bundle["model"].predict_proba(x)[0, 1])
    if max(p, 1 - p) < t["min_conf"]:
        return dict(verdict="Unreliable", p_fake=p, reason=f"classifier not confident (P(fake) = {p:.2f})")
    return dict(verdict="Fake" if p >= 0.5 else "Real", p_fake=p, reason=f"P(fake) = {p:.2f}")


def analyze_npz(npz, model_path=MODEL_PATH, **thr):
    f, _ = feat.video_features(npz)
    if f is None:
        return dict(verdict="Unreliable", reason="clip shorter than 5 s", p_fake=None, features=None)
    out = decide(f, load_model(model_path), **thr)
    out["features"] = f
    out["npz"] = str(npz)
    return out


def analyze_video(video, backend="mediapipe", model_path=MODEL_PATH, name=None, max_seconds=None, **thr):
    from extract_signal import process_video
    video = Path(video)
    name = name or video.stem
    try:
        process_video(video, "unknown", backend, backend, max_seconds, None, "pos", False, name, False)
    except SystemExit as e:
        return dict(verdict="Unreliable", reason=str(e), p_fake=None, features=None)
    return analyze_npz(ROOT / "cache" / "signals" / f"{name}__{backend}.npz", model_path, **thr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "haar"])
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    r = analyze_video(a.video, a.backend, max_seconds=a.max_seconds)
    print(f"\nVERDICT: {r['verdict']}\nREASON : {r['reason']}")
