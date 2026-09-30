"""
CoolGuard - core decision logic (no Streamlit dependency)

Kept separate from coolguard_app.py so it can be imported and tested without
pulling in the UI, and so the exact same logic could later run on a phone
app, a supervisor dashboard, or directly on the ESP32 side if needed.
"""

import math

# ---------------------------------------------------------------------------
# WBGT ESTIMATE
# Simplified shade-only approximation (no solar radiation term), adapted for
# a low-cost sensor setup. Cite the real WBGT standard (ISO 7243) in your
# report and note this simplification explicitly.
# ---------------------------------------------------------------------------
def estimate_wbgt(temp_c: float, humidity_pct: float) -> float:
    e = (humidity_pct / 100) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return round(0.567 * temp_c + 0.393 * e + 3.94, 1)


def wbgt_risk(wbgt: float) -> tuple:
    if wbgt < 28:
        return 0, "Low"
    if wbgt < 30:
        return 1, "Moderate"
    if wbgt < 32:
        return 2, "High"
    return 3, "Extreme"


def hr_risk(resting_hr: int, current_hr: int, age: int) -> tuple:
    max_hr_est = 220 - age
    pct_of_max = current_hr / max_hr_est * 100
    rise = current_hr - resting_hr
    if pct_of_max >= 85 or rise >= 40:
        return 2, "High strain"
    if pct_of_max >= 70 or rise >= 25:
        return 1, "Elevated"
    return 0, "Normal"


def bmi_note(weight_kg: float, height_cm: float) -> str:
    h_m = height_cm / 100
    bmi = weight_kg / (h_m ** 2)
    tag = " (higher body mass can raise heat-strain risk)" if bmi >= 30 else ""
    return f"BMI \u2248 {bmi:.1f}{tag}."


SYMPTOM_WEIGHTS = {
    "None": 0, "Mild thirst": 1, "Heavy sweating": 1, "Tiredness / weakness": 2,
    "Headache": 2, "Dizziness": 3, "Nausea": 3, "Muscle cramps": 3,
    "Confusion": 5, "Not sweating despite heat": 5, "Rapid heartbeat / palpitations": 3,
}
DANGER_SYMPTOMS = {"Confusion", "Not sweating despite heat"}


def symptom_score(selected: list) -> int:
    if "None" in selected or not selected:
        return 0
    return sum(SYMPTOM_WEIGHTS.get(s, 0) for s in selected)


def rest_minutes(decision: str, score: int) -> int:
    if decision == "\U0001F534 STOP":
        return min(30, 15 + score * 2)
    if decision == "\U0001F7E1 REST SOON":
        return 8
    return 0


def plausibility_check(wbgt: float, skin_temp: float, resting_hr: int, current_hr: int):
    """Flags readings where the environment sensor and the body's own signals
    disagree, a sign of possible tampering (e.g. a heat source held near the
    sensor) or a sensor fault, rather than a real heat-stress event."""
    hr_rise = current_hr - resting_hr
    if wbgt >= 32 and skin_temp < 37.0 and hr_rise < 10:
        return ("Ambient heat reading is extreme, but skin temperature and heart rate look "
                "normal. Flagged for supervisor review rather than an automatic STOP.")
    return None


def decide(wbgt: float, resting_hr: int, current_hr: int, age: int, symptoms: list,
           new_worker: bool = False):
    """Combines all signals into one decision. Returns (decision, score, rest_min)."""
    w_level, _ = wbgt_risk(wbgt)
    h_level, _ = hr_risk(resting_hr, current_hr, age)
    s_score = symptom_score(symptoms)

    score = w_level + h_level + (s_score // 2)
    if age >= 50 or new_worker:
        score += 1

    if s_score >= 5 or score >= 5:
        decision = "\U0001F534 STOP"
    elif score >= 3:
        decision = "\U0001F7E1 REST SOON"
    else:
        decision = "\U0001F7E2 CONTINUE"

    return decision, score, rest_minutes(decision, score)
