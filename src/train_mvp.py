"""
train_mvp.py - train the real-vs-fake classifier on cache/features.csv

    python src/train_mvp.py                      # needs BOTH real (0) and fake (1) videos
    python src/train_mvp.py --gate-by bpp

What it does
  * compares a BASELINE (one global pulse) with the FULL feature set (per-region + agreement),
    each with logistic regression and random forest
  * evaluation never mixes a group (identity) between train and test:
      - if features.csv has a `split` column with train/test -> train on train, score on test
      - otherwise grouped, stratified cross-validation (out-of-fold predictions)
  * AUC with a 95% bootstrap interval (resampling videos), confusion matrix,
    and a coverage-vs-accuracy table for the video-quality gate
  * saves models/mvp.joblib for predict.py and the demo
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from features import FEATURE_SETS  # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced", random_state=0))


MODELS = {"logreg": make_lr, "forest": make_rf}


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["label"] in ("0", "1")]
    if not rows:
        sys.exit("No labelled videos (label 0 = real, 1 = fake) in features.csv")
    y = np.array([int(r["label"]) for r in rows])
    if len(set(y)) < 2:
        sys.exit("Only one class present. Training needs BOTH real and fake videos.")
    return rows, y, np.array([r["group"] for r in rows]), np.array([r.get("split", "") for r in rows])


def matrix(rows, cols):
    return np.array([[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in cols] for r in rows])


def predict_oof(make, X, y, groups, split, seed=0):
    """Returns (indices evaluated, probability of fake). Groups are never shared across folds."""
    if "train" in set(split) and "test" in set(split):
        tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
        m = make().fit(X[tr], y[tr])
        return te, m.predict_proba(X[te])[:, 1]
    k = min(5, len({g for g, c in zip(groups, y) if c == 0}), len({g for g, c in zip(groups, y) if c == 1}))
    if k < 2:
        sys.exit("Need at least 2 separate groups (identities) per class for cross-validation.")
    idx, prob = [], []
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        m = make().fit(X[tr], y[tr])
        idx += list(te); prob += list(m.predict_proba(X[te])[:, 1])
    return np.array(idx), np.array(prob)


def auc_ci(y, p, n=1000, seed=0):
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); a = roc_auc_score(y, p); bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], p[i]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_training(features_csv, model_out, gate_by="bpp", plot_path=None, min_conf=0.6, quiet=False):
    rows, y, groups, split = load(features_csv)
    say = (lambda *a: None) if quiet else print
    say(f"{len(y)} videos: {int((y == 0).sum())} real, {int((y == 1).sum())} fake, {len(set(groups))} groups")
    if len(y) < 20:
        say("WARNING: fewer than 20 videos. Treat every number below as a smoke test, not a result.")

    say("\nmodel     features   AUC    95% CI")
    best = None
    for fs_name, cols in FEATURE_SETS.items():
        X = matrix(rows, cols)
        for m_name, make in MODELS.items():
            idx, p = predict_oof(make, X, y, groups, split)
            a, lo, hi = auc_ci(y[idx], p)
            say(f"{m_name:9s} {fs_name:9s} {a:5.3f}  [{lo:.3f}, {hi:.3f}]")
            key = (a, fs_name == "full", m_name == "logreg")        # ties -> full, then logreg
            if best is None or key > best[0]:
                best = (key, fs_name, m_name, idx, p)
    _, fs_name, m_name, idx, p = best
    say(f"\nchosen: {m_name} on '{fs_name}' features")
    cm = confusion_matrix(y[idx], (p >= 0.5).astype(int), labels=[0, 1])
    say("confusion matrix at 0.5   (rows = truth real/fake, cols = predicted real/fake)")
    say(cm)

    # coverage vs accuracy: drop the lowest-quality videos by the gate variable
    q = np.array([float(rows[i][gate_by]) if rows[i][gate_by] not in ("", "nan") else np.nan for i in idx])
    cov, acc = [], []
    say(f"\nquality gate on '{gate_by}': remove the lowest X% of videos")
    say("removed  coverage  accuracy  AUC")
    for pct in (0, 10, 20, 30, 40, 50):
        thr = np.nanpercentile(q, pct) if pct and not np.all(np.isnan(q)) else -np.inf
        keep = np.isnan(q) | (q >= thr) if pct else np.ones(len(q), bool)
        if keep.sum() == 0:
            continue
        a_ = float(np.mean((p[keep] >= 0.5).astype(int) == y[idx][keep]))
        auc_k = roc_auc_score(y[idx][keep], p[keep]) if len(set(y[idx][keep])) == 2 else float("nan")
        cov.append(keep.mean()); acc.append(a_)
        say(f"{pct:5d}%   {keep.mean():6.0%}    {a_:6.1%}   {auc_k:.3f}")
    if plot_path:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.5)); plt.plot(np.array(cov) * 100, np.array(acc) * 100, "o-")
        plt.xlabel(f"coverage (% of videos answered, gate = {gate_by})"); plt.ylabel("accuracy on answered (%)")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(plot_path, dpi=130); plt.close()

    final = MODELS[m_name]().fit(matrix(rows, FEATURE_SETS[fs_name]), y)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=final, features=FEATURE_SETS[fs_name], feature_set=fs_name, model_name=m_name,
                     oof_auc=best[0][0], min_conf=min_conf, gate_by=gate_by), model_out)
    say(f"\nSaved {model_out}")
    return dict(auc=best[0][0], feature_set=fs_name, model=m_name, n=len(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "cache" / "features.csv"))
    ap.add_argument("--out", default=str(ROOT / "models" / "mvp.joblib"))
    ap.add_argument("--gate-by", default="bpp", choices=["bpp", "detected_frac", "q_best_snr", "motion_rel"])
    ap.add_argument("--plot", default=str(ROOT / "cache" / "coverage_accuracy.png"))
    a = ap.parse_args()
    run_training(a.features, a.out, a.gate_by, a.plot)
