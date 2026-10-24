import re, sys, csv, argparse
from collections import Counter
from pathlib import Path
sys.path.insert(0, "src")
from huggingface_hub import snapshot_download
import extract_signal as ex

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
root = Path(snapshot_download("NetherlandsForensicInstitute/rppg-deepfake-detection", repo_type="dataset"))
vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov"))
rel = lambda p: p.relative_to(root)
print(Counter(rel(p).parts[:2] for p in vids))            # check real vs fake folders here
real = [p for p in vids if rel(p).parts[0].lower() == "real"]
fake = [p for p in vids if rel(p).parts[0].lower() != "real"]
if a.limit:
    real, fake = real[:a.limit // 2], fake[:a.limit // 2]
print(f"{len(real)} real, {len(fake)} fake")

rows = []
for lab, group in ((0, real), (1, fake)):
    for p in group:
        name = "_".join(rel(p).with_suffix("").parts)
        m = re.search(r"id\d+", p.stem)
        out = Path("cache/signals") / f"{name}__hf.npz"
        if not out.exists():
            try:
                ex.process_video(p, "fake" if lab else "real", "hf", "mediapipe", 10,
                                 None, "pos", False, name, False)
            except SystemExit as e:
                print("skip", name, e); continue
        rows.append(dict(name=name, label=lab, group=m.group(0) if m else name, split=""))
with open("data/labels_hf.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "label", "group", "split"])
    w.writeheader(); w.writerows(rows)
print("wrote data/labels_hf.csv with", len(rows), "videos")