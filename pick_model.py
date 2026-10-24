import glob, sys, time, numpy as np, rppg
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1      # how many clips to use
clips = [(v, v[:-4] + "_gt.txt") for v in sorted(glob.glob("data/ubfc_test/*.avi"))][:N]
names = ["ME-chunk.rlap", "PhysMamba.rlap", "RhythmMamba.rlap", "EfficientPhys.rlap", "TSCAN.rlap"]
def true_hr(p): return float(np.loadtxt(p)[1].mean())   # heart-rate row, whole clip
for n in names:
    t0 = time.time()
    try:
        m = rppg.Model(n); e = []; q = []
        for v, g in clips:
            r = m.process_video(v)
            e.append(abs(r["hr"] - true_hr(g))); q.append(r.get("SQI", float("nan")))
        print(f"RESULT {n:20s} MAE {np.mean(e):5.1f} bpm  SQI {np.nanmean(q):.2f}  {len(e)} clips  {time.time()-t0:.0f}s", flush=True)
    except Exception as ex:
        print("RESULT", n, "failed:", str(ex)[:80], flush=True)