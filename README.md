# PulseProof (Hashcode 14.0)

Stress-testing heartbeat-based (rPPG) deepfake detection, with a reliability score
that lets the detector answer Real / Fake / **Unreliable**.

## Setup
    python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
    pip install -r requirements.txt

If `mediapipe==0.10.14` fails to install, delete that line: the code falls back to OpenCV Haar automatically.

## Quick checks
    python src/selftest.py                                   # pulse maths works, no video needed
    python src/extract_signal.py data/me.mp4 --label real    # video -> signals + plot

## Good test video
One person, face large and still, even light (near a window), 20-30 s, no backlight.

## Output contract (shared with teammates)
`cache/signals/{video}__{cond}.npz` with `rgb_forehead`, `rgb_lcheek`, `rgb_rcheek` (T x 3),
`fps`, `label` (real=0, fake=1, unknown=-1), `detected_frac`.
