import sys, numpy as np
sys.path.insert(0, "src"); import rppg

def run(npz, gt_path):
    d = np.load(npz); fps = float(d["fps"])
    gt = np.loadtxt(gt_path); hr, t = gt[1], gt[2]
    W, S = int(10*fps), int(5*fps)
    for name in ("forehead", "lcheek", "rcheek"):
        x = d["rgb_" + name]; errs = []; snrs = []
        for a in range(0, len(x) - W, S):
            p = rppg.extract_pulse(x[a:a+W], fps)
            m = (t >= a/fps) & (t < (a+W)/fps)
            if m.sum() == 0: continue
            errs.append(abs(rppg.peak_bpm(p, fps) - hr[m].mean()))
            snrs.append(rppg.snr_db(p, fps))
        print(f"  {name:9s} MAE {np.mean(errs):5.1f} bpm  SNR {np.mean(snrs):5.1f} dB  ({len(errs)} windows)")

for tag, sub in [("s1_mp","subject1"),("s1_haar","subject1"),("s2_mp","subject2"),("s2_haar","subject2")]:
    print(tag); run(f"cache/signals/vid__{tag}.npz", f"data/ubfc/{sub}/ground_truth.txt")