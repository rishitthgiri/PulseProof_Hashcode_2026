"""
selftest.py - checks the pulse maths WITHOUT any video.
Builds fake face colours with a hidden 72 bpm heartbeat plus noise, and checks
that rppg.py recovers it.   Run:  python src/selftest.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rppg  # noqa: E402
from extract_signal import fill_missing  # noqa: E402

rng = np.random.default_rng(0)
fs, T, true_bpm = 30.0, 600, 72.0
t = np.arange(T) / fs
pulse = np.sin(2 * np.pi * (true_bpm / 60) * t)

def fake_region(noise):
    base = np.array([180.0, 120.0, 100.0])
    amp = np.array([0.0015, 0.0045, 0.0020]) * base          # green strongest, like real skin
    drift = 1 + 0.02 * np.sin(2 * np.pi * 0.05 * t)[:, None]  # slow lighting drift
    x = base * drift + np.outer(pulse, amp) + rng.normal(0, noise, (T, 3))
    return x

regions = {"forehead": fake_region(0.05), "lcheek": fake_region(0.08), "rcheek": fake_region(0.08)}
res = rppg.analyze_regions(regions, fs)
ok = True
for k in regions:
    err = abs(res["bpm"][k] - true_bpm)
    print(f"{k:9s} bpm={res['bpm'][k]:.1f}  SNR={res['snr_db'][k]:.1f} dB  err={err:.1f}")
    ok &= err < 2.0
print("coherence:", {k: round(v, 2) for k, v in res["coherence"].items()})

# pure noise must give a LOW SNR (this is what the reliability score will rely on)
noise_only = rppg.extract_pulse(rng.normal(150, 1.0, (T, 3)), fs)
print(f"noise-only SNR = {rppg.snr_db(noise_only, fs):.1f} dB (should be well below the real ones)")
ok &= rppg.snr_db(noise_only, fs) < res["snr_db"]["forehead"] - 3

# missing-frame filling keeps one box per frame
b = np.full((10, 3, 4), np.nan); b[3] = 1; b[7] = 2
f = fill_missing(b)
ok &= f is not None and f.shape == b.shape and not np.isnan(f).any() and f[0, 0, 0] == 1 and f[5, 0, 0] == 1 and f[9, 0, 0] == 2
print("fill_missing OK" if ok else "SOMETHING FAILED")
sys.exit(0 if ok else 1)
