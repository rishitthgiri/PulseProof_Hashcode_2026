import numpy as np, rppg
clips = [("data/ubfc_test/s1.avi", "data/ubfc_test/s1_gt.txt"),
         # add 4 or more pairs
        ]
names = ['PhysNet.rlap','PhysNet.pure','TSCAN.rlap','TSCAN.pure','EfficientPhys.rlap',
         'EfficientPhys.pure','PhysFormer.rlap','PhysFormer.pure','RhythmMamba.rlap',
         'PhysMamba.rlap','ME-chunk.rlap','ME-flow.rlap']
def true_hr(p, secs=20):
    g = np.loadtxt(p); return float(g[1][g[2] < secs].mean())
for n in names:
    try:
        m = rppg.Model(n); e = []
        for v, g in clips:
            e.append(abs(m.process_video(v)["hr"] - true_hr(g)))
        print(f"{n:20s} MAE {np.mean(e):5.1f} bpm over {len(e)} clips")
    except Exception as ex:
        print(n, "failed:", str(ex)[:80])