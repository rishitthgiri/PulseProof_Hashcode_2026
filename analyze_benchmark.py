"""
analyze_benchmark.py - reads cache/benchmark1_windows.csv (made by run_benchmark.py).
Shows: worst subjects, how often each region is within 5 bpm, whether picking the
best-SNR region per window helps, and how error falls when low-SNR windows are dropped.
    python analyze_benchmark.py
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parent / "cache" / "benchmark1_windows.csv"
rows = list(csv.DictReader(open(path)))
for r in rows:
    for k in ("start_s", "abs_err", "snr_db"):
        r[k] = float(r[k])

print("== forehead MAE per subject (best first) ==")
by = defaultdict(list)
for r in rows:
    if r["region"] == "forehead":
        by[r["subject"]].append(r["abs_err"])
for s, e in sorted(by.items(), key=lambda kv: np.mean(kv[1])):
    print(f"  {s:12s} {np.mean(e):6.1f} bpm")

print("\n== windows within 5 bpm of the truth, per region ==")
for reg in ("forehead", "lcheek", "rcheek"):
    e = np.array([r["abs_err"] for r in rows if r["region"] == reg])
    print(f"  {reg:9s} {100 * np.mean(e < 5):4.0f}%   median error {np.median(e):.1f} bpm")

win = defaultdict(list)
for r in rows:
    win[(r["subject"], r["start_s"])].append(r)
best = [max(v, key=lambda x: x["snr_db"]) for v in win.values()]
e = np.array([b["abs_err"] for b in best]); snr = np.array([b["snr_db"] for b in best])
print(f"\n== best-SNR region per window: median error {np.median(e):.1f} bpm, within 5 bpm {100 * np.mean(e < 5):.0f}%")

print("\n== keep only windows whose best SNR >= threshold ==")
print("  thresh  coverage  median_err  mean_err  within5bpm")
for th in (-99, -4, -2, 0, 2, 4, 6):
    k = snr >= th
    if k.sum():
        print(f"  {th:5}   {100 * k.mean():5.0f}%    {np.median(e[k]):6.1f}    {e[k].mean():6.1f}    {100 * np.mean(e[k] < 5):5.0f}%")
