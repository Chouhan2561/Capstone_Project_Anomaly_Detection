import time

import torch
import torch.nn as nn

from src import config as C
from src.data import prepare
from src.model import LSTMAE


def run(device, X, epochs=3, batch=128):
    model = LSTMAE(len(C.FEATURES), C.HIDDEN, C.LATENT, C.WINDOW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss_fn = nn.MSELoss()
    X = X.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), batch):
            b = X[perm[i:i + batch]]
            opt.zero_grad()
            loss_fn(model(b), b).backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / epochs


if __name__ == "__main__":
    X = torch.tensor(prepare(save_scaler=False)["W_train"])
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
    print("CPU  seconds per epoch: %.2f" % run("cpu", X))
    if torch.cuda.is_available():
        run("cuda", X, epochs=1)                          # warm-up, not timed
        print("GPU, batch 128, seconds per epoch: %.2f" % run("cuda", X))
        print("GPU, batch 512, seconds per epoch: %.2f" % run("cuda", X, batch=512))