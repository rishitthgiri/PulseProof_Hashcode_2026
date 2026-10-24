import importlib

packages = [
    "cv2",
    "mediapipe",
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "matplotlib",
    "streamlit",
]

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "version unavailable")
        print(f"[OK] {package}: {version}")
    except Exception as exc:
        print(f"[FAIL] {package}: {exc}")

try:
    import cv2
    import mediapipe
    import numpy
    import scipy
    import sklearn
    import pandas
    import matplotlib
    import streamlit

    print("\nEnvironment imports completed.")
except Exception:
    raise SystemExit("One or more required imports failed.")