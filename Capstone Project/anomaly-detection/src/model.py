import torch
import torch.nn as nn


class LSTMAE(nn.Module):
    """LSTM autoencoder: squeeze a window into a short summary, then rebuild it."""

    def __init__(self, n_feat=5, hidden=64, latent=16, window=10):
        super().__init__()
        self.window = window
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)    # encoder
        self.to_lat = nn.Linear(hidden, latent)                 # latent features
        self.from_lat = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)    # decoder
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)                    # read the whole window
        z = self.to_lat(h[-1])                     # squeeze into latent features
        d = self.from_lat(z).unsqueeze(1).repeat(1, self.window, 1)
        o, _ = self.dec(d)                         # rebuild the window
        return self.out(o)


if __name__ == "__main__":
    m = LSTMAE()
    x = torch.randn(4, 10, 5)
    print("input :", tuple(x.shape))
    print("output:", tuple(m(x).shape))
    print("parameters:", sum(p.numel() for p in m.parameters()))