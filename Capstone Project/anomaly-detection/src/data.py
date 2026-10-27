import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src import config as C


def load_data():
    df = pd.read_csv(C.DATA_PATH)
    if C.EXTRA_FEATURES:
        df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
        df["Temp gap [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
        df["Wear x torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


def make_windows(X, y, L):
    """Cut a table into overlapping windows of L consecutive rows.
    A window is labelled 1 (anomalous) if ANY row inside it is a failure."""
    n = len(X) - L + 1
    W = np.stack([X[i:i + L] for i in range(n)]).astype("float32")
    labels = np.array([y[i:i + L].max() for i in range(n)])
    return W, labels


def prepare(save_scaler=True):
    df = load_data()
    X = df[C.FEATURES].values.astype("float32")
    y = df[C.LABEL].values

    # 1) Split by row order: 70% train, 15% validation, 15% test
    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)
    parts = {"train": slice(0, i1), "val": slice(i1, i2), "test": slice(i2, n)}

    # 2) Scaler learns ONLY from normal training rows (prevents data leakage)
    tr = parts["train"]
    scaler = StandardScaler().fit(X[tr][y[tr] == 0])
    Xs = scaler.transform(X).astype("float32")

    # 3) Windows are made inside each split, so no window crosses a boundary
    windows = {name: make_windows(Xs[sl], y[sl], C.WINDOW) for name, sl in parts.items()}

    data = {
        "W_train": windows["train"][0][windows["train"][1] == 0],      # normal only
        "W_val_normal": windows["val"][0][windows["val"][1] == 0],     # for the threshold
        "W_test": windows["test"][0],
        "l_test": windows["test"][1],
        "scaler": scaler,
        "rows": {k: (v.start, v.stop) for k, v in parts.items()},
        "failure_rate_rows": float(y.mean()),
        "windows_total": {k: len(v[0]) for k, v in windows.items()},
    }
    if save_scaler:
        joblib.dump(scaler, C.SCALER_PATH)
    return data


if __name__ == "__main__":
    d = prepare()
    print("Rows per split:", d["rows"])
    print("Failure rate in the rows: %.2f%%" % (100 * d["failure_rate_rows"]))
    print("Windows made per split:", d["windows_total"])
    print("W_train (normal only):", d["W_train"].shape)
    print("W_val_normal:", d["W_val_normal"].shape)
    print("W_test:", d["W_test"].shape,
          "| anomalous windows: %d (%.1f%%)" % (d["l_test"].sum(), 100 * d["l_test"].mean()))
    print("Scaled train mean (should be near 0): %.3f" % d["W_train"].mean())
    print("Scaled train std  (should be near 1): %.3f" % d["W_train"].std())
    print("Scaler saved to:", C.SCALER_PATH)