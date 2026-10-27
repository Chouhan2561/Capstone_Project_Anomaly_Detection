import copy
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src import config as C
from src.data import prepare
from src.detect import get_errors
from src.model import LSTMAE


def train_one(seed, d, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xtr = torch.tensor(d["W_train"]).to(device)
    Xva = torch.tensor(d["W_val_normal"]).to(device)
    model = LSTMAE(len(C.FEATURES), C.HIDDEN, C.LATENT, C.WINDOW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss_fn = nn.MSELoss()
    best, wait, best_state, stopped = float("inf"), 0, None, C.EPOCHS
    for epoch in range(C.EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), C.BATCH):
            b = Xtr[perm[i:i + C.BATCH]]
            opt.zero_grad(); loss_fn(model(b), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), Xva).item()
        if v < best - 1e-6:
            best, wait, best_state = v, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= C.PATIENCE:
                stopped = epoch
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), best, stopped


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    seeds = [0, 1, 2, 3, 4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = prepare(save_scaler=False)
    y = d["l_test"]
    rows = []
    for s in seeds:
        model, best, stopped = train_one(s, d, device)
        err_val, _, _ = get_errors(model, d["W_val_normal"])
        err_te, _, _ = get_errors(model, d["W_test"])
        thr = np.percentile(err_val, C.PERCENTILE)
        p = (err_te > thr).astype(int)
        rows.append({"seed": s, "stopped_at": stopped, "val_loss": round(best, 4),
                     "precision": precision_score(y, p, zero_division=0),
                     "recall": recall_score(y, p, zero_division=0),
                     "f1": f1_score(y, p, zero_division=0),
                     "roc_auc": roc_auc_score(y, err_te),
                     "pr_auc": average_precision_score(y, err_te)})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.round(3).to_csv(f"outputs/seeds_{tag}.csv", index=False)
    cols = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    print(f"\n=== {tag}: mean and standard deviation over {len(seeds)} seeds ({device}) ===")
    print(pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std()}).round(3).to_string())


if __name__ == "__main__":
    main()