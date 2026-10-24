"""
merge_fallback.py - merge pulse features + stitching/skin features + heart-rate fallback.

    python src/merge_fallback.py                                  # without open-rppg
    python src/merge_fallback.py --orp cache/openrppg.csv         # with open-rppg

openrppg.csv columns: name,bpm,sqi   (one row per video; make it from open-rppg output)
Fallback per video:
  1. open-rppg if bpm in 40..180 and sqi >= --sqi-thr          -> hr_src = 0
  2. else POS if best_snr >= --snr-thr and n_agree >= 2         -> hr_src = 1
  3. else no pulse (NaN heart rate)                             -> hr_src = 2
Set --sqi-thr on UBFC (validation), never on the fakes.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--pulse", default=str(ROOT / "cache" / "features.csv"))
ap.add_argument("--extra", default=str(ROOT / "cache" / "extra_features.csv"))
ap.add_argument("--orp", default=None)
ap.add_argument("--sqi-thr", type=float, default=0.5)
ap.add_argument("--snr-thr", type=float, default=0.0)
ap.add_argument("--out", default=str(ROOT / "cache" / "features_all.csv"))
a = ap.parse_args()

d = pd.read_csv(a.pulse).merge(pd.read_csv(a.extra), on="name", how="inner")
if a.orp:
    o = pd.read_csv(a.orp)[["name", "bpm", "sqi"]].rename(columns={"bpm": "orp_bpm", "sqi": "orp_sqi"})
    d = d.merge(o, on="name", how="left")
else:
    d["orp_bpm"], d["orp_sqi"] = np.nan, np.nan
orp_ok = d.orp_bpm.between(40, 180) & (d.orp_sqi >= a.sqi_thr)
pos_ok = (d.best_snr >= a.snr_thr) & (d.n_agree >= 2)
d["hr_src"] = np.where(orp_ok, 0, np.where(pos_ok, 1, 2))
d["hr_bpm"] = np.where(orp_ok, d.orp_bpm, np.where(pos_ok, d.g_bpm, np.nan))
d["fallback_used"] = (d.hr_src == 1).astype(int)
d["hr_gap"] = (d.orp_bpm - d.g_bpm).abs()
d["group"] = d["group"].astype(str)
d.to_csv(a.out, index=False)
print(f"{len(d)} videos -> {a.out}")
print("heart-rate source: open-rppg", int((d.hr_src == 0).sum()), "| POS fallback", int((d.hr_src == 1).sum()),
      "| none", int((d.hr_src == 2).sum()))
