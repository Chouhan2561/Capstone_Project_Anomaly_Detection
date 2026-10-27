import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.data import prepare
from src.model import LSTMAE


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main():
    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    d = prepare()
    Xtr = torch.tensor(d["W_train"]).to(device)
    Xva = torch.tensor(d["W_val_normal"]).to(device)

    model = LSTMAE(len(C.FEATURES), C.HIDDEN, C.LATENT, C.WINDOW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss_fn = nn.MSELoss()

    best, wait, history = float("inf"), 0, []
    for epoch in range(C.EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        batch_losses = []
        for i in range(0, len(Xtr), C.BATCH):
            b = Xtr[perm[i:i + C.BATCH]]
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), Xva).item()
        history.append((float(np.mean(batch_losses)), val))
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d}  train {history[-1][0]:.4f}  val {val:.4f}")

        if val < best - 1e-6:
            best, wait = val, 0
            torch.save(model.state_dict(), C.MODEL_PATH)
        else:
            wait += 1
            if wait >= C.PATIENCE:
                print("Early stopping at epoch", epoch)
                break

    json.dump({"window": C.WINDOW, "hidden": C.HIDDEN, "latent": C.LATENT},
              open(C.CONFIG_PATH, "w"))

    plt.plot([h[0] for h in history], label="train")
    plt.plot([h[1] for h in history], label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.legend()
    plt.savefig("outputs/loss_curve.png", dpi=150)
    print("Best validation loss: %.4f" % best)
    print("Saved:", C.MODEL_PATH, "and outputs/loss_curve.png")


if __name__ == "__main__":
    main()