"""PulseProof demo.   Run:  streamlit run app.py"""
import sys, time
from pathlib import Path
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import predict, rppg  # noqa: E402

st.set_page_config(page_title="PulseProof", layout="wide")
st.title("PulseProof: can we trust the pulse?")
st.caption("Decision support only. A human must review before anyone's video is called fake.")

backend = st.sidebar.selectbox("Face detector", ["mediapipe", "haar"])
min_bpp = st.sidebar.slider("Min. compression quality (bits/pixel)", 0.0, 0.2, 0.02, 0.005)
min_conf = st.sidebar.slider("Min. classifier confidence", 0.5, 0.95, 0.6, 0.05)
if predict.load_model() is None:
    st.sidebar.warning("No trained model yet (models/mvp.joblib). The demo shows pulse and quality only.")

up = st.file_uploader("Upload a face video (one person, face visible, 10 s or more)", type=["mp4", "avi", "mov"])
if up and st.button("Analyze"):
    dst = ROOT / "cache" / "uploads"; dst.mkdir(parents=True, exist_ok=True)
    path = dst / f"{int(time.time())}_{up.name}"; path.write_bytes(up.getbuffer())
    with st.spinner("Extracting pulse ..."):
        out = predict.analyze_video(path, backend, name=path.stem, min_bpp=min_bpp, min_conf=min_conf)
    colour = {"Real": "green", "Fake": "red", "Unreliable": "orange", "Untrained": "gray"}[out["verdict"]]
    st.markdown(f"## Verdict: :{colour}[{out['verdict']}]")
    st.write(out["reason"])
    f = out.get("features")
    if f:
        c = st.columns(4)
        c[0].metric("Best-region SNR (dB)", f"{f['q_best_snr']:.1f}")
        c[1].metric("Region bpm spread", f"{f['bpm_spread']:.1f}")
        c[2].metric("Face found", f"{f['detected_frac']:.0%}")
        c[3].metric("Bits / pixel", f"{f['bpp']:.3f}")
        d = np.load(out["npz"]); fps = float(d["fps"])
        res = rppg.analyze_regions({r: d["rgb_" + r] for r in ("forehead", "lcheek", "rcheek")}, fps)
        png = ROOT / "cache" / "uploads" / f"{path.stem}.png"
        rppg.plot_report(res, fps, path.name, png)
        st.image(str(png))
