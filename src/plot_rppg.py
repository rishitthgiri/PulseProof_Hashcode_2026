import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="outputs/vision_cardio")
    args = parser.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = data["time_seconds"]
    wave = data["rppg_value"]

    if len(wave) < 4 or not np.isfinite(wave).all():
        raise SystemExit("Waveform is too short or contains invalid values.")

    fs = 6.4
    centered = wave - wave.mean()
    windowed = centered * np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(windowed), d=1 / fs)

    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not band.any():
        raise SystemExit("No frequencies in the configured pulse band.")

    peak_idx = np.flatnonzero(band)[np.argmax(spectrum[band])]
    bpm = freqs[peak_idx] * 60

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, wave)
    ax.set(title="Predicted rPPG waveform", xlabel="Time (seconds)",
           ylabel="Model output")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "my_video_waveform.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs * 60, spectrum)
    ax.axvline(bpm, linestyle="--", label=f"Peak: {bpm:.1f} BPM")
    ax.set(xlim=(40, 180), title="rPPG frequency spectrum",
           xlabel="Frequency (BPM)", ylabel="Power")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "my_video_spectrum.png", dpi=160)
    plt.close(fig)

    print(f"Waveform standard deviation: {wave.std():.6f}")
    print(f"Waveform range: {wave.min():.6f} to {wave.max():.6f}")
    print(f"Spectral peak: {bpm:.1f} BPM")
    print(f"Saved plots in: {out}")


if __name__ == "__main__":
    main()
