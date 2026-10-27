import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data import prepare
from src.detect import get_errors, load_model
from src.explain import explain

st.set_page_config(page_title="Machine anomaly monitor", layout="wide")


@st.cache_resource(show_spinner="Loading data and model...")
def load_all():
    cfg = json.load(open(C.CONFIG_PATH))
    d = prepare(save_scaler=False)
    model = load_model()
    err, ps, R = get_errors(model, d["W_test"])
    return cfg, d, err, ps, R


cfg, d, err, ps, R = load_all()
if "threshold" not in cfg:
    st.error("No threshold saved yet. Run: python -m src.detect")
    st.stop()

W, y, scaler = d["W_test"], d["l_test"], d["scaler"]

st.title("Industrial machine anomaly monitor")
st.caption("LSTM autoencoder detects. A local LLM only explains. "
           "This is an early warning, not a confirmed fault.")

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
thr = st.sidebar.slider("Alert threshold (reconstruction error)", 0.0,
                        float(np.percentile(err, 99.9)), float(cfg["threshold"]), step=0.001,
                        format="%.3f")
flagged = np.where(err > thr)[0]
st.sidebar.write(f"Flagged **{len(flagged)}** of {len(err)} windows")
only_flagged = st.sidebar.checkbox("Browse flagged windows only", value=True)
pool = flagged if (only_flagged and len(flagged)) else np.arange(len(err))
pos = st.sidebar.slider("Window", 0, max(len(pool) - 1, 1), 0) if len(pool) > 1 else 0
i = int(pool[min(pos, len(pool) - 1)])
use_llm = st.sidebar.checkbox("Use local LLM (Ollama)", value=True)

# ---------------- headline numbers ----------------
status = "Possible Anomaly" if err[i] > thr else "Normal"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Window", i)
c2.metric("Anomaly score", f"{err[i]:.4f}")
c3.metric("Threshold", f"{thr:.4f}")
c4.metric("Status", status)
st.caption("Ground truth from the dataset (shown for the demo only): "
           + ("this window contains a failure." if y[i] == 1 else "no failure inside this window."))

# ---------------- error graph ----------------
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(err, lw=0.7, label="reconstruction error")
ax.axhline(thr, color="black", ls="--", label="threshold")
bad = np.where(y == 1)[0]
ax.scatter(bad, err[bad], s=6, color="red", alpha=0.4, label="windows containing a failure")
ax.axvline(i, color="green", lw=1.5, label="selected window")
ax.set_xlabel("test window"); ax.set_ylabel("error"); ax.legend(loc="upper right", fontsize=8)
st.pyplot(fig)

# ---------------- original vs rebuilt ----------------
orig, rec = scaler.inverse_transform(W[i]), scaler.inverse_transform(R[i])
n = len(C.NAMES); cols = min(n, 4); rows = math.ceil(n / cols)
fig2, axes = plt.subplots(rows, cols, figsize=(12, 2.4 * rows), squeeze=False)
for j in range(rows * cols):
    a = axes[j // cols][j % cols]
    if j < n:
        a.plot(orig[:, j], label="original"); a.plot(rec[:, j], "--", label="rebuilt")
        a.set_title(C.NAMES[j], fontsize=10)
        if j == 0:
            a.legend(fontsize=8)
    else:
        a.axis("off")
fig2.tight_layout()
st.subheader("Original vs rebuilt sensor readings")
st.pyplot(fig2)

# ---------------- explanation ----------------
st.subheader("Why was this flagged?")
if status == "Normal":
    st.info("This window is below the threshold, so there is nothing to explain.")
else:
    key = f"exp_{i}_{use_llm}_{thr:.4f}"
    if st.button("Explain this alert"):
        with st.spinner("Asking the local model... (can take up to a minute on a CPU)"):
            text, source, sensors = explain(W[i], err[i], ps[i], scaler, thr, use_llm)
        st.session_state[key] = (text, source, sensors)
    if key in st.session_state:
        text, source, sensors = st.session_state[key]
        st.table(pd.DataFrame([{"Sensor": s["name"], "Value": round(s["value"], 1),
                                "Normal average": round(s["normal"], 1), "Std from normal": round(s["z"], 1),
                                "Share of error": f"{100 * s['share']:.0f}%"} for s in sensors]))
        st.write(text)
        st.caption(f"Source: {source}. A suggestion for a human to check, not a diagnosis.")

# ---------------- saved results ----------------
with st.expander("Saved evaluation results"):
    for f, title in [("outputs/model_comparison.csv", "Model comparison"),
                     ("outputs/threshold_sweep.csv", "Effect of the threshold rule")]:
        try:
            st.write(title); st.dataframe(pd.read_csv(f))
        except Exception:
            st.write(f"({f} not found. Run python -m src.detect)")