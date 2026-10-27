# All settings live here, so you can change them in one place.
SEED = 42

DATA_PATH = "data/ai4i2020.csv"
EXTRA_FEATURES = True    # set False to use only the 5 raw sensors

BASE_FEATURES = ["Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
BASE_NAMES = ["Air temp", "Process temp", "Speed", "Torque", "Tool wear"]
BASE_UNITS = ["K", "K", "rpm", "Nm", "min"]

if EXTRA_FEATURES:
    FEATURES = BASE_FEATURES + ["Power [W]", "Temp gap [K]", "Wear x torque"]
    NAMES = BASE_NAMES + ["Power", "Temp gap", "Wear x torque"]
    UNITS = BASE_UNITS + ["W", "K", "min*Nm"]
else:
    FEATURES, NAMES, UNITS = BASE_FEATURES, BASE_NAMES, BASE_UNITS
LABEL = "Machine failure"

# Windows
WINDOW = 10

# Model
HIDDEN = 64
LATENT = 16
EPOCHS = 300
PATIENCE = 8
BATCH = 128
LR = 1e-3

# Threshold
PERCENTILE = 95

# Saved files
MODEL_PATH = "models/model.pt"
SCALER_PATH = "models/scaler.pkl"
CONFIG_PATH = "models/config.json"

# Local LLM
OLLAMA_MODEL = "llama3.2"