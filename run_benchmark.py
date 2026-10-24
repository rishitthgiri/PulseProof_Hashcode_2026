"""
run_benchmark.py - Benchmark 1: run the extractor on EVERY UBFC subject and compare
with the ground-truth heart rate (10 s windows, 5 s step).

Run from the repo root (the folder that contains src/ and data/):
    python run_benchmark.py --root data/archive --backend mediapipe --limit 2    # quick test first
    python run_benchmark.py --root data/archive --backend mediapipe              # all subjects

Safe to stop and re-run: subjects already extracted are skipped (use --force to redo).
Outputs:
    cache/benchmark1_windows.csv   one row per subject / region / window
    cache/benchmark1_subjects.csv  one row per subject / region (MAE, mean SNR)
"""
import argparse, csv, re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pulse as rppg
from extract_signal import process_video, ROOT

REGIONS = ("forehead", "lcheek", "rcheek")


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def find_subjects(root):
    out = []
    for d in sorted((p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: natural(p.name)):
        vid = next((f for f in sorted(d.rglob("*")) if f.suffix.lower() in (".avi", ".mp4")), None)
        gt = next((f for f in sorted(d.rglob("*")) if f.name.lower().startswith("ground_truth")), None)
        if vid and gt:
            out.append((d.name, vid, gt))
        else:
            print(f"skipping {d.name}: needs a video and a ground_truth file")
    return out


def extract(name, vid, backend, max_seconds, force):
    out = ROOT / "cache" / "signals" / f"{name}__{backend}.npz"
    if out.exists() and not force:
        print(f"[{name}] already extracted, skipping")
        return out
    try:
        process_video(vid, label="real", cond=backend, backend=backend,
                      max_seconds=max_seconds, name=name, plot=False)
    except SystemExit as e:                          # bad video etc: skip, keep going
        print(f"[{name}] SKIPPED: {e}")
        return None
    return out


def evaluate(npz, gt_path, subject, win_s=10, step_s=5):
    d = np.load(npz)
    fps = float(d["fps"])
    gt = np.loadtxt(gt_path)
    if gt.shape[0] != 3 and gt.shape[1] == 3:
        gt = gt.T
    hr, t = gt[1], gt[2]                              # rows: PPG, heart rate, time (s)
    W, S = int(win_s * fps), int(step_s * fps)
    rows = []
    for region in REGIONS:
        x = d["rgb_" + region]
        for a in range(0, len(x) - W + 1, S):
            m = (t >= a / fps) & (t < (a + W) / fps)
            if m.sum() == 0:
                continue
            p = rppg.extract_pulse(x[a:a + W], fps)
            est, snr, g = rppg.peak_bpm(p, fps), rppg.snr_db(p, fps), float(hr[m].mean())
            rows.append(dict(subject=subject, region=region, start_s=round(a / fps, 1),
                             est_bpm=round(est, 1), gt_bpm=round(g, 1),
                             abs_err=round(abs(est - g), 2), snr_db=round(snr, 2)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/archive", help="folder containing subject1, subject3, ...")
    ap.add_argument("--backend", choices=["mediapipe", "haar"], default="mediapipe")
    ap.add_argument("--limit", type=int, default=None, help="only the first N subjects")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    subs = find_subjects(a.root)
    if a.limit:
        subs = subs[:a.limit]
    print(f"{len(subs)} subjects found in {a.root}")
    if not subs:
        sys.exit("No subjects found. Check --root (should contain subject1, subject3, ...)")

    all_rows, t0 = [], time.time()
    for i, (name, vid, gt) in enumerate(subs, 1):
        print(f"\n=== {i}/{len(subs)}  {name}   ({(time.time() - t0) / 60:.1f} min elapsed) ===")
        npz = extract(name, vid, a.backend, a.max_seconds, a.force)
        if npz is None:
            continue
        rows = evaluate(npz, gt, name)
        all_rows += rows
        for r in REGIONS:
            e = [x["abs_err"] for x in rows if x["region"] == r]
            s = [x["snr_db"] for x in rows if x["region"] == r]
            if e:
                print(f"  {r:9s} MAE {np.mean(e):5.1f} bpm   mean SNR {np.mean(s):5.1f} dB")

    if not all_rows:
        sys.exit("No results.")
    cache = ROOT / "cache"
    with open(cache / "benchmark1_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)

    names = sorted({r["subject"] for r in all_rows}, key=natural)
    summ = []
    for n in names:
        for r in REGIONS:
            e = [x["abs_err"] for x in all_rows if x["subject"] == n and x["region"] == r]
            s = [x["snr_db"] for x in all_rows if x["subject"] == n and x["region"] == r]
            if e:
                summ.append(dict(subject=n, region=r, mae_bpm=round(np.mean(e), 2),
                                 mean_snr_db=round(np.mean(s), 2), windows=len(e)))
    with open(cache / "benchmark1_subjects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    print("\n================ SUMMARY (per-subject MAE, 10 s windows) ================")
    for r in REGIONS:
        m = np.array([x["mae_bpm"] for x in summ if x["region"] == r])
        print(f"{r:9s} subjects {len(m):2d} | median MAE {np.median(m):5.1f} | mean {m.mean():5.1f} | "
              f"under 5 bpm: {(m < 5).sum()}/{len(m)} | under 10 bpm: {(m < 10).sum()}/{len(m)}")
    pos = [x["abs_err"] for x in all_rows if x["snr_db"] > 0]
    neg = [x["abs_err"] for x in all_rows if x["snr_db"] <= 0]
    print(f"\nwindows with SNR > 0 dB : {len(pos):5d}  mean error {np.mean(pos) if pos else float('nan'):5.1f} bpm")
    print(f"windows with SNR <= 0 dB: {len(neg):5d}  mean error {np.mean(neg) if neg else float('nan'):5.1f} bpm")
    print(f"\nSaved: cache/benchmark1_windows.csv and cache/benchmark1_subjects.csv")


if __name__ == "__main__":
    main()
