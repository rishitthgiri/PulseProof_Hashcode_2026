"""
train_all.py - ablation: pulse only vs stitching only vs pulse+stitching (+ fallback features),
then a skin-tone fairness table.      python src/train_all.py
Skin tone (ITA) is NOT a classifier input; it is only used to report per-group behaviour.
"""
import sys
from pathlib import Path
import pandas as pd
SRC = Path(__file__).resolve().parent; ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import features as feat, train_mvp as tm

df = pd.read_csv(ROOT / "cache" / "features_all.csv")
STITCH = [c for c in df.columns if c.startswith("st_") and c != "st_face_frac"]
FB = [c for c in ("hr_src", "hr_gap", "orp_sqi") if c in df.columns and df[c].notna().any()]
base = dict(feat.FEATURE_SETS)
feat.FEATURE_SETS.clear()          # same dict object train_mvp uses
feat.FEATURE_SETS.update({"baseline": base["baseline"], "pulse": base["full"], "stitch": STITCH,
                          "pulse+stitch": base["full"] + STITCH})
if FB:
    feat.FEATURE_SETS["pulse+stitch+fb"] = base["full"] + STITCH + FB
tm.run_training(str(ROOT / "cache" / "features_all.csv"), str(ROOT / "models" / "mvp.joblib"),
                gate_by="detected_frac", plot_path=str(ROOT / "cache" / "coverage_accuracy.png"))

bins = [-90, -30, 10, 28, 41, 55, 90]
names = ["dark", "brown", "tan", "intermediate", "light", "very light"]
df["tone"] = pd.cut(df.ita, bins=bins, labels=names)
print("\nSkin-tone buckets (ITA proxy). Small counts = not enough to conclude anything.")
print(df.groupby("tone", observed=True).agg(n=("name", "size"), pct_fake=("label", "mean"),
      median_best_snr=("best_snr", "median"), pct_no_pulse=("hr_src", lambda s: (s == 2).mean())).round(2))
