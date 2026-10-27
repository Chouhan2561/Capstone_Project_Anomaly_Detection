# Industrial Machine Anomaly Detection (LSTM Autoencoder + Local LLM Explainer)

Learns normal machine behaviour from sensor data and flags unusual windows
as an early warning. A local LLM (Ollama, no API key) explains each alert.

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI, synthetic):
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Place `ai4i2020.csv` in `data/`.

## Setup (Python 3.12)
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
Install Ollama from ollama.com, then: `ollama pull llama3.2`

## Run, in order
    python -m src.data      # prepare data, windows, scaler
    python -m src.train     # train the LSTM autoencoder
    python -m src.detect    # threshold, metrics, baseline comparison
    python -m src.explain --n 3
    streamlit run app.py    # dashboard

## How it works
1. Scale the 5 sensors using normal training rows only; cut into windows of 10 rows.
2. Train an LSTM autoencoder on normal windows only.
3. Error per window = how badly it is rebuilt. Above the threshold = Possible Anomaly.
4. Top sensors per alert come from per-sensor error.
5. A local LLM writes a short explanation, and simple checks reject unsafe wording.

## Limitations
- The data is synthetic and has no real timestamps, so windows are an assumption.
- Failures are rare (3.4% of rows), and one failing row marks up to 10 windows.
- The threshold is a trade-off between false alarms and missed failures.
- The LLM can give wrong or exaggerated explanations. Checks catch only some problems.
- Early warning only. It does not guarantee failure prevention.