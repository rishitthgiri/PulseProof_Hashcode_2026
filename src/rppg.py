"""
rppg.py - pure signal-processing helpers (no video code here).

RGB signal (T x 3, one row per frame)  ->  pulse wave  ->  heart rate / SNR / coherence
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, periodogram, coherence

LOW_HZ, HIGH_HZ = 0.7, 3.0          # 42 - 180 bpm


# ----------------------------------------------------------------- filtering
def bandpass(x, fs, lo=LOW_HZ, hi=HIGH_HZ, order=4):
    """Zero-phase Butterworth bandpass (filtfilt => no phase shift, so
    phase lags between face regions stay meaningful)."""
    nyq = 0.5 * fs
    hi = min(hi, 0.95 * nyq)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    if len(x) < 40:                                 # too short to filter safely
        return np.asarray(x, dtype=float) - np.mean(x)
    return sosfiltfilt(sos, x)


# ------------------------------------------------------------- pulse methods
def pos(rgb, fs, win_sec=1.6):
    """POS (Wang et al. 2017), sliding window + overlap-add. rgb: (T,3)."""
    rgb = np.asarray(rgb, dtype=float)
    T = len(rgb)
    L = int(round(win_sec * fs))
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    H = np.zeros(T)
    if T < L or L < 8:
        return H
    for n in range(L, T + 1):
        m = n - L
        C = rgb[m:n].T                               # (3, L)
        mu = C.mean(axis=1, keepdims=True)
        if np.any(mu <= 1e-6):
            continue
        S = P @ (C / mu)                             # (2, L)
        h = S[0] + (S[0].std() / (S[1].std() + 1e-8)) * S[1]
        H[m:n] += h - h.mean()
    return H


def chrom(rgb, fs, win_sec=1.6):
    """CHROM (de Haan & Jeanne 2013), sliding window + overlap-add. rgb: (T,3)."""
    rgb = np.asarray(rgb, dtype=float)
    T = len(rgb)
    L = int(round(win_sec * fs))
    H = np.zeros(T)
    if T < L or L < 8:
        return H
    for n in range(L, T + 1):
        m = n - L
        C = rgb[m:n].T
        mu = C.mean(axis=1, keepdims=True)
        if np.any(mu <= 1e-6):
            continue
        R, G, B = C / mu
        X = 3 * R - 2 * G
        Y = 1.5 * R + G - 1.5 * B
        h = X + (X.std() / (Y.std() + 1e-8)) * Y
        H[m:n] += h - h.mean()
    return H


def extract_pulse(rgb, fs, method="pos"):
    raw = pos(rgb, fs) if method == "pos" else chrom(rgb, fs)
    return bandpass(raw, fs)


# ------------------------------------------------------------------ spectrum
def spectrum(pulse, fs, lo=LOW_HZ, hi=HIGH_HZ, nfft=8192):
    f, p = periodogram(pulse - np.mean(pulse), fs=fs, window="hann", nfft=nfft)
    m = (f >= lo) & (f <= hi)
    return f[m], p[m]


def peak_bpm(pulse, fs):
    f, p = spectrum(pulse, fs)
    return float(f[np.argmax(p)] * 60.0)


def snr_db(pulse, fs, tol_hz=0.1):
    """De Haan-style SNR: power around the peak (and its 2nd harmonic) vs the
    rest of the 0.7-3 Hz band, in dB. Higher = cleaner pulse."""
    f, p = spectrum(pulse, fs)
    f0 = f[np.argmax(p)]
    sig = (np.abs(f - f0) <= tol_hz) | (np.abs(f - 2 * f0) <= tol_hz)
    s, n = p[sig].sum(), p[~sig].sum()
    return float(10 * np.log10((s + 1e-12) / (n + 1e-12)))


def band_coherence(a, b, fs):
    """Mean magnitude-squared coherence of two pulses inside 0.7-3 Hz (0..1)."""
    # need several averaged segments, otherwise coherence is trivially ~1.0
    nper = int(min(len(a) // 4, 4 * fs))
    if nper < 32:
        return float("nan")
    f, c = coherence(a, b, fs=fs, nperseg=nper)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(np.mean(c[m]))


def analyze_regions(rgb_by_region, fs, method="pos"):
    """rgb_by_region: {'forehead': (T,3), 'lcheek': ..., 'rcheek': ...}
    Returns pulses + per-region bpm/SNR + pairwise coherence + bpm spread."""
    pulses = {k: extract_pulse(v, fs, method) for k, v in rgb_by_region.items()}
    out = {
        "pulses": pulses,
        "bpm": {k: peak_bpm(v, fs) for k, v in pulses.items()},
        "snr_db": {k: snr_db(v, fs) for k, v in pulses.items()},
        "coherence": {},
    }
    keys = list(pulses)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            out["coherence"][f"{keys[i]}-{keys[j]}"] = band_coherence(
                pulses[keys[i]], pulses[keys[j]], fs)
    b = list(out["bpm"].values())
    out["bpm_spread"] = float(max(b) - min(b))
    return out


# ---------------------------------------------------------------------- plot
def plot_report(res, fs, title, save_path=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res["pulses"])
    colors = {"forehead": "crimson", "lcheek": "tab:blue", "rcheek": "tab:green"}
    fig, ax = plt.subplots(len(names) + 1, 1, figsize=(10, 2.2 * (len(names) + 1)))
    for a, k in zip(ax, names):
        p = res["pulses"][k]
        a.plot(np.arange(len(p)) / fs, p, color=colors.get(k), lw=1.2)
        a.set_ylabel(k)
        a.set_title(f"{k}: {res['bpm'][k]:.0f} bpm, SNR {res['snr_db'][k]:.1f} dB",
                    fontsize=9, loc="left")
        a.grid(alpha=.3, ls="--")
    ax[len(names) - 1].set_xlabel("time (s)")
    for k in names:
        f, p = spectrum(res["pulses"][k], fs)
        ax[-1].plot(f * 60, p / (p.max() + 1e-12), color=colors.get(k), label=k)
    ax[-1].set_xlabel("heart rate (bpm)")
    ax[-1].set_ylabel("norm. power")
    ax[-1].legend(fontsize=8)
    ax[-1].grid(alpha=.3, ls="--")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
