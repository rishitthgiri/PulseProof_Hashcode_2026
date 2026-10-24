"""
features.py - signals (.npz from extract_signal.py)  ->  features per window and per video.

    python src/features.py --cond mediapipe                       # labels taken from the .npz files
    python src/features.py --cond mediapipe --labels data/labels.csv

labels.csv (optional) columns:  name,label,group,split
    name   = video name as used in cache/signals/{name}__{cond}.npz
    label  = real / fake   (or 0 / 1)
    group  = identity or source-video id. Videos sharing a group are NEVER split between
             train and test (a real video and the fakes made from it share one group)
    split  = train / test  (optional; if present it is used instead of cross-validation)

Outputs: cache/features.csv (one row per video), cache/features_windows.csv (one row per window)
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import pulse as rppg  # noqa: E402

REGIONS = ("forehead", "lcheek", "rcheek")
PAIRS = (("forehead", "lcheek"), ("forehead", "rcheek"), ("lcheek", "rcheek"))

# baseline = what a single-global-pulse detector would see
BASELINE = ["g_bpm", "g_snr", "g_pr", "g_ent"]
# full = baseline + per-region quality + agreement between regions
FULL = (BASELINE
        + [f"{k}_{r}" for r in REGIONS for k in ("snr", "pr", "ent")]
        + ["coh_mean", "coh_min", "bpm_spread", "bpm_std", "best_snr", "n_agree"])
QUALITY = ["bpp", "detected_frac", "longest_gap_s", "motion_rel", "duration_s", "n_windows", "q_best_snr"]
FEATURE_SETS = {"baseline": BASELINE, "full": FULL}


def _window_row(win, fps):
    pulses = {r: rppg.extract_pulse(win[r], fps) for r in REGIONS}
    row, bpms = {}, []
    for r in REGIONS:
        p = pulses[r]
        row[f"snr_{r}"] = rppg.snr_db(p, fps)
        row[f"pr_{r}"] = rppg.peak_ratio(p, fps)
        row[f"ent_{r}"] = rppg.spec_entropy(p, fps)
        bpms.append(rppg.peak_bpm(p, fps))
    g = rppg.extract_pulse(np.mean([win[r] for r in REGIONS], axis=0), fps)   # one global pulse
    row.update(g_bpm=rppg.peak_bpm(g, fps), g_snr=rppg.snr_db(g, fps),
               g_pr=rppg.peak_ratio(g, fps), g_ent=rppg.spec_entropy(g, fps))
    cohs = [rppg.band_coherence(pulses[a], pulses[b], fps) for a, b in PAIRS]
    row["coh_mean"] = float(np.nanmean(cohs)) if not np.all(np.isnan(cohs)) else float("nan")
    row["coh_min"] = float(np.nanmin(cohs)) if not np.all(np.isnan(cohs)) else float("nan")
    row["bpm_spread"] = float(max(bpms) - min(bpms))
    row["bpm_std"] = float(np.std(bpms))
    row["best_snr"] = max(row[f"snr_{r}"] for r in REGIONS)
    row["n_agree"] = float(sum(abs(b - np.median(bpms)) <= 5 for b in bpms))
    return row


def window_features(rgb, fps, win_s=10.0, step_s=5.0):
    """rgb: {'forehead': (T,3), ...} -> list of per-window feature dicts."""
    T = len(rgb["forehead"])
    W, S = int(win_s * fps), int(step_s * fps)
    if T < int(5 * fps):
        return []
    if T < W:
        starts = [0]; W = T                    # short clip: one window over everything
    else:
        starts = range(0, T - W + 1, S)
    rows = []
    for a in starts:
        r = _window_row({k: v[a:a + W] for k, v in rgb.items()}, fps)
        r["start_s"] = round(a / fps, 2)
        rows.append(r)
    return rows


def video_features(npz_path, win_s=10.0, step_s=5.0):
    """-> (feature dict for the whole video, list of window rows). (None, []) if too short."""
    d = np.load(npz_path)
    fps = float(d["fps"])
    rgb = {r: d["rgb_" + r].astype(float) for r in REGIONS}
    rows = window_features(rgb, fps, win_s, step_s)
    if not rows:
        return None, []
    f = {k: float(np.nanmean([r[k] for r in rows])) for k in FULL}
    f["q_best_snr"] = float(np.median([r["best_snr"] for r in rows]))
    f["n_windows"] = float(len(rows))
    f["duration_s"] = len(rgb["forehead"]) / fps
    for k, default in (("detected_frac", 1.0), ("longest_gap_s", 0.0), ("bpp", float("nan")), ("motion_rel", float("nan"))):
        f[k] = float(d[k]) if k in d.files else default
    f["_label_in_npz"] = int(d["label"]) if "label" in d.files else -1
    return f, rows


def _read_labels(path):
    out = {}
    if path:
        for r in csv.DictReader(open(path)):
            lab = {"real": 0, "fake": 1, "0": 0, "1": 1}.get(str(r["label"]).strip().lower(), -1)
            out[r["name"]] = (lab, r.get("group") or r["name"], (r.get("split") or "").strip().lower())
    return out


def build(signals_dir, cond, labels_csv=None, out_dir=None, quiet=False):
    signals_dir = Path(signals_dir)
    out_dir = Path(out_dir or signals_dir.parent)
    labels = _read_labels(labels_csv)
    vrows, wrows = [], []
    for npz in sorted(signals_dir.glob(f"*__{cond}.npz")):
        name = npz.name[: -len(f"__{cond}.npz")]
        f, rows = video_features(npz)
        if f is None:
            if not quiet: print(f"skip {name}: clip shorter than 5 s")
            continue
        lab, grp, split = labels.get(name, (f["_label_in_npz"], name, ""))
        vrows.append(dict(name=name, label=lab, group=grp, split=split,
                          **{k: f[k] for k in FULL + QUALITY + []}))
        for r in rows:
            wrows.append(dict(name=name, label=lab, group=grp, **r))
    if not vrows:
        sys.exit(f"No *__{cond}.npz files in {signals_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn, rows in (("features.csv", vrows), ("features_windows.csv", wrows)):
        with open(out_dir / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if not quiet:
        labs = [r["label"] for r in vrows]
        print(f"{len(vrows)} videos ({labs.count(0)} real, {labs.count(1)} fake, {labs.count(-1)} unlabelled)")
        print(f"Saved {out_dir / 'features.csv'} and features_windows.csv")
    return vrows, wrows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="mediapipe", help="tag after __ in cache/signals file names")
    ap.add_argument("--signals", default=str(ROOT / "cache" / "signals"))
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    build(a.signals, a.cond, a.labels)
