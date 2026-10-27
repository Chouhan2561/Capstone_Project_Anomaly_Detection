import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src import config as C
from src.data import prepare
from src.model import LSTMAE


def load_model():
    m = LSTMAE(len(C.FEATURES), C.HIDDEN, C.LATENT, C.WINDOW)
    m.load_state_dict(torch.load(C.MODEL_PATH, map_location="cpu"))
    m.eval()
    return m


def reconstruct(model, W):
    out = []
    with torch.no_grad():
        for i in range(0, len(W), 512):
            out.append(model(torch.tensor(W[i:i + 512])).numpy())
    return np.concatenate(out)


def get_errors(model, W):
    """Returns: error per window, error per sensor, rebuilt windows."""
    R = reconstruct(model, W)
    per_sensor = ((W - R) ** 2).mean(axis=1)          # shape (N, 5)
    return per_sensor.mean(axis=1), per_sensor, R


def scores(name, y, score, pred):
    return {"model": name,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score)}


def main():
    d = prepare(save_scaler=False)
    W_train, W_val, W_te, y_te = d["W_train"], d["W_val_normal"], d["W_test"], d["l_test"]

    model = load_model()
    err_val, _, _ = get_errors(model, W_val)
    err_te, ps_te, _ = get_errors(model, W_te)

    # Threshold = a high percentile of the errors on NORMAL validation windows
    threshold = float(np.percentile(err_val, C.PERCENTILE))
    pred = (err_te > threshold).astype(int)           # 1 = Possible Anomaly
    print("Threshold (%.1fth percentile of normal val errors): %.4f" % (C.PERCENTILE, threshold))
    print("Flagged %d of %d test windows" % (pred.sum(), len(pred)))
    print("Random-guess PR-AUC would be about %.3f (the anomaly rate)" % y_te.mean())

    # Baseline: Isolation Forest on flattened windows
    flat = lambda W: W.reshape(len(W), -1)
    iso = IsolationForest(n_estimators=200, random_state=C.SEED).fit(flat(W_train))
    s_iso = -iso.score_samples(flat(W_te))
    t_iso = np.percentile(-iso.score_samples(flat(W_val)), C.PERCENTILE)
    rows = [scores("LSTM-AE", y_te, err_te, pred),
            scores("IsolationForest", y_te, s_iso, (s_iso > t_iso).astype(int))]
    table = pd.DataFrame(rows).round(3)
    print("\n=== Model comparison (test set) ===")
    print(table.to_string(index=False))
    print("\nConfusion matrix for LSTM-AE  [[TN FP] [FN TP]]")
    print(confusion_matrix(y_te, pred))

    # How the threshold choice changes the result
    sweep = []
    for label, t in [("p90", np.percentile(err_val, 90)), ("p95", np.percentile(err_val, 95)),
                     ("p99", np.percentile(err_val, 99)), ("p99.5", np.percentile(err_val, 99.5)),
                     ("mean+3std", err_val.mean() + 3 * err_val.std())]:
        p = (err_te > t).astype(int)
        sweep.append({"rule": label, "threshold": round(float(t), 4), "flagged": int(p.sum()),
                      "precision": precision_score(y_te, p, zero_division=0),
                      "recall": recall_score(y_te, p, zero_division=0),
                      "f1": f1_score(y_te, p, zero_division=0)})
    sweep = pd.DataFrame(sweep).round(3)
    print("\n=== Effect of the threshold rule ===")
    print(sweep.to_string(index=False))

    # Save results and the threshold for later steps
    table.to_csv("outputs/model_comparison.csv", index=False)
    sweep.to_csv("outputs/threshold_sweep.csv", index=False)
    cfg = json.load(open(C.CONFIG_PATH))
    cfg.update({"threshold": threshold, "percentile": C.PERCENTILE})
    json.dump(cfg, open(C.CONFIG_PATH, "w"))

    plt.figure(figsize=(14, 4))
    plt.plot(err_te, lw=0.8, label="reconstruction error")
    plt.axhline(threshold, color="black", ls="--", label="threshold")
    bad = np.where(y_te == 1)[0]
    plt.scatter(bad, err_te[bad], color="red", s=8, label="windows containing a failure")
    plt.xlabel("test window"); plt.ylabel("error"); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/error_plot.png", dpi=150)
    print("\nSaved: outputs/error_plot.png, model_comparison.csv, threshold_sweep.csv")


if __name__ == "__main__":
    main()