import csv, numpy as np
rows = list(csv.DictReader(open("cache/features.csv")))
cols = ["best_snr", "g_snr", "bpm_spread", "n_agree", "detected_frac", "duration_s"]
for lab, name in (("0", "real"), ("1", "fake")):
    r = [x for x in rows if x["label"] == lab]
    print(f"\n{name} (n={len(r)})")
    for c in cols:
        v = np.array([float(x[c]) for x in r if x[c] not in ("", "nan")])
        print(f"  {c:14s} median {np.median(v):6.2f}   min {v.min():6.2f}   max {v.max():6.2f}")