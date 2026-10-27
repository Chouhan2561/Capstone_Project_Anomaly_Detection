import json
import sys
import time

import numpy as np

from src import config as C
from src.data import prepare
from src.detect import get_errors, load_model

UNITS = C.UNITS
ALIASES = {
    "Air temp": ["air temp"],
    "Process temp": ["process temp"],
    "Speed": ["speed", "rpm"],
    "Torque": ["torque"],
    "Tool wear": ["tool wear"],
    "Power": ["power"],
    "Temp gap": ["temp gap", "temperature gap", "temperature difference"],
    "Wear x torque": ["wear", "torque"],
}
BANNED = ["motor", "gearbox", "bearing", "lubricat", "misalign", "imbalance",
          "vibration", "pump", "belt", "coolant leak", "electrical"]
MAX_WORDS = 90


def top_sensors(window, per_sensor_err, scaler, k=3):
    """Top sensors by error. Shows each sensor's MOST EXTREME row in the window
    (the window is already in z-score units), not the window average."""
    idx = np.argsort(per_sensor_err)[::-1][:k]
    share = per_sensor_err / per_sensor_err.sum()
    out = []
    for j in idx:
        r = int(np.argmax(np.abs(window[:, j])))
        z = float(window[r, j])
        val = float(scaler.inverse_transform(window[r:r + 1])[0, j])
        out.append({"name": C.NAMES[j], "value": val, "normal": float(scaler.mean_[j]),
                    "unit": UNITS[j], "z": z, "share": float(share[j])})
    return out


def level(z):
    a = abs(z)
    return "within the usual range" if a < 2 else ("unusually far from normal" if a < 3 else "very far from normal")


def build_prompt(err, thr, sensors):
    lines = "\n".join(
        f"- {s['name']}: most extreme reading in the window {s['value']:.1f} {s['unit']} "
        f"(normal about {s['normal']:.1f}; {abs(s['z']):.1f} standard deviations "
        f"{'above' if s['z'] > 0 else 'below'} normal, {level(s['z'])}; "
        f"{100 * s['share']:.0f}% of the error)" for s in sensors)
    return f"""You are a maintenance assistant. An anomaly detector flagged a Possible Anomaly.
Reconstruction error: {err:.2f} (threshold {thr:.2f}).
Top sensors by error:
{lines}

Rules:
- Use ONLY the data above. Do not invent numbers.
- Call a sensor abnormal only if it is 2 or more standard deviations from normal.
  If none are, say the detector flagged an unusual combination and no single sensor is clearly abnormal.
- Do not use words like significant, serious, critical or severe.
- Mention no machine parts except: the cutting tool, cooling, drive power, or load.
- Write at most 60 words, in 2 to 3 sentences.
- End by saying this is an early warning, not a confirmed fault."""


def template_explanation(sensors):
    parts = "; ".join(f"{s['name']} is {s['value']:.0f} {s['unit']} (normal about {s['normal']:.0f})"
                      for s in sensors)
    return (f"Possible anomaly: {parts}. Please inspect the machine. "
            "This is an early warning, not a confirmed fault.")


def verify(text, sensors):
    """Simple faithfulness checks. Returns (ok, reason)."""
    t = text.lower()
    names = [s["name"] for s in sensors]
    for w in BANNED:
        if w in t:
            return False, f"mentions '{w}', which was not in the data"
    if len(text.split()) > MAX_WORDS:
        return False, "too long"
    if not any(a in t for a in ALIASES[sensors[0]["name"]]):
        return False, f"does not name the strongest sensor ({sensors[0]['name']})"
    if "early warning" not in t:
        return False, "missing the early-warning statement"
    max_z = max(abs(s["z"]) for s in sensors)
    if max_z < 3 and any(w in t for w in ["significant", "serious", "critical", "severe"]):
        return False, "exaggerates a small deviation"
    says_none = "no single sensor" in t or "none of the sensors" in t
    if max_z < 2 and not (says_none or "combination" in t):
        return False, "does not say that no single sensor is clearly abnormal"
    if max_z >= 2 and says_none:
        return False, "says nothing is abnormal although a sensor is 2+ std away"
    if "cutting tool" in t and not any(n in names for n in ["Tool wear", "Wear x torque"]):
        return False, "blames the cutting tool but tool wear is not a top sensor"
    if "cooling" in t and not any(n in names for n in ["Process temp", "Air temp"]):
        return False, "mentions cooling but no temperature sensor is a top sensor"
    return True, "ok"


def ask_llm(prompt, tries=2):
    try:
        import ollama
    except Exception:
        return None
    for attempt in range(tries):
        try:
            r = ollama.chat(model=C.OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}],
                            options={"temperature": 0.2})
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"   (Ollama problem: {type(e).__name__}, attempt {attempt + 1})")
            time.sleep(3)
    return None


def explain(window, err, per_sensor_err, scaler, threshold, use_llm=True):
    sensors = top_sensors(window, per_sensor_err, scaler)
    status = "template"
    if use_llm:
        for _ in range(2):
            text = ask_llm(build_prompt(err, threshold, sensors))
            if text is None:
                break
            ok, reason = verify(text, sensors)
            if ok:
                return text, "LLM (verified)", sensors
            status = f"template (LLM rejected: {reason})"
    return template_explanation(sensors), status, sensors


def main():
    use_llm = "--no-llm" not in sys.argv
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3

    cfg = json.load(open(C.CONFIG_PATH))
    if "threshold" not in cfg:
        raise SystemExit("No threshold saved yet. Run: python -m src.detect")
    threshold = cfg["threshold"]

    d = prepare(save_scaler=False)
    scaler, W, y = d["scaler"], d["W_test"], d["l_test"]
    err, ps, _ = get_errors(load_model(), W)
    flagged = np.where(err > threshold)[0]
    print(f"Threshold {threshold:.4f} | flagged {len(flagged)} of {len(W)} test windows")
    if len(flagged) == 0:
        raise SystemExit("Nothing flagged. Lower PERCENTILE in config.py and rerun detect.")

    hits = [i for i in flagged if y[i] == 1][:n]
    falses = [i for i in flagged if y[i] == 0][:max(1, n // 2)]
    for i in hits + falses:
        text, source, sensors = explain(W[i], err[i], ps[i], scaler, threshold, use_llm)
        kind = "real failure window" if y[i] == 1 else "FALSE ALARM"
        print("\n" + "=" * 70)
        print(f"Window {i} | error {err[i]:.3f} | {kind}")
        for s in sensors:
            print(f"   {s['name']}: {s['value']:.1f} {s['unit']} (normal {s['normal']:.1f}, "
                  f"{s['z']:+.1f} std, {100 * s['share']:.0f}% of error)")
        print(f"[{source}] {text}")


if __name__ == "__main__":
    main()