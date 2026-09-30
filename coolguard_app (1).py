"""
CoolGuard - Heat stress & dehydration early-warning app for outdoor workers
Innovators of Tomorrow 2026 (Qatar Scientific Club x WISH) - Innovation track

Run:   pip install streamlit pandas plotly
       streamlit run coolguard_app.py

IMPORTANT: Educational/safety-awareness tool only. NOT a medical device.
Blood glucose and blood pressure are NOT included: no low-cost wearable
sensor can measure these accurately without a cuff or blood draw.
"""

import base64
import io
import math
import struct
import sys
import wave
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, ".")
from coolguard_core import (
    estimate_wbgt, wbgt_risk, hr_risk, symptom_score, rest_minutes,
    plausibility_check, bmi_note, SYMPTOM_WEIGHTS, DANGER_SYMPTOMS,
)

st.set_page_config(page_title="CoolGuard", page_icon="🥵", layout="wide")

try:
    import joblib
    _ML_MODEL = joblib.load("coolguard_model.joblib")
except Exception:
    _ML_MODEL = None


def ml_risk_probability(wbgt, current_hr, resting_hr, age, moisture):
    if _ML_MODEL is None:
        return None
    hr_rise = current_hr - resting_hr
    return float(_ML_MODEL.predict_proba([[wbgt, hr_rise, age, moisture]])[0][1])


# ---------------------------------------------------------------------------
# THEME / CUSTOM STYLING
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --cg-bg: #0B2B3C; --cg-card: #ffffff; --cg-accent: #17A398; --cg-accent-dark: #0E6B62;
  --cg-warn: #F2A93C; --cg-danger: #E15554; --cg-safe: #3FA66D; --cg-text: #0B2B3C;
}
.cg-card {
  background: var(--cg-card); border-radius: 14px; padding: 18px 20px;
  box-shadow: 0 2px 10px rgba(11,43,60,0.08); border: 1px solid rgba(11,43,60,0.06);
  margin-bottom: 10px;
}
.cg-card h4 { margin: 0 0 4px 0; font-size: 13px; color: #6b7c85; font-weight: 500; }
.cg-card .cg-value { font-size: 26px; font-weight: 700; color: var(--cg-text); }
.cg-card .cg-sub { font-size: 12px; color: #8a97a0; }
.cg-badge { display:inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; }
.cg-badge-safe { background: rgba(63,166,109,0.15); color: var(--cg-safe); }
.cg-badge-warn { background: rgba(242,169,60,0.18); color: #8a5a10; }
.cg-badge-danger { background: rgba(225,85,84,0.15); color: var(--cg-danger); }
.cg-hero { text-align:center; padding: 40px 10px 20px; }
.cg-hero h1 { font-size: 40px; margin-bottom: 4px; }
.cg-hero p { color: #6b7c85; font-size: 16px; }
</style>
""", unsafe_allow_html=True)


def card(title, value, sub="", icon=""):
    st.markdown(f"""
    <div class="cg-card"><h4>{icon} {title}</h4>
    <div class="cg-value">{value}</div><div class="cg-sub">{sub}</div></div>
    """, unsafe_allow_html=True)


def badge(text, kind):
    cls = {"safe": "cg-badge-safe", "warn": "cg-badge-warn", "danger": "cg-badge-danger"}[kind]
    st.markdown(f'<span class="cg-badge {cls}">{text}</span>', unsafe_allow_html=True)


def beep():
    """Generates a short beep tone on the fly (no external audio file needed)
    and plays it via an autoplaying HTML5 audio tag."""
    framerate, duration, freq = 44100, 0.35, 880
    n = int(framerate * duration)
    buf = io.BytesIO()
    wf = wave.open(buf, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(framerate)
    for i in range(n):
        val = int(32767 * 0.35 * math.sin(2 * math.pi * freq * i / framerate))
        wf.writeframesraw(struct.pack("<h", val))
    wf.close()
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" '
                'type="audio/wav"></audio>', unsafe_allow_html=True)


def render_gauge(risk_percent: float, key: str):
    color = "#3FA66D" if risk_percent < 40 else "#F2A93C" if risk_percent < 70 else "#E15554"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": "Heat stress risk", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40], "color": "rgba(63,166,109,0.15)"},
                {"range": [40, 70], "color": "rgba(242,169,60,0.15)"},
                {"range": [70, 100], "color": "rgba(225,85,84,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_trend(rows: list, key: str):
    if len(rows) < 2:
        st.caption("Trend chart appears after at least 2 readings for this worker.")
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["wbgt"], mode="lines+markers", name="WBGT (°C)",
                             line=dict(color="#E15554")))
    fig.add_trace(go.Scatter(y=df["current_hr"], mode="lines+markers", name="Heart rate (bpm)",
                             line=dict(color="#17A398"), yaxis="y2"))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="WBGT (°C)"), yaxis2=dict(title="HR (bpm)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
for k, d in [("log", []), ("connected", False), ("device_name", None), ("rest_until", {}),
             ("entered", False), ("demo_step", 0)]:
    if k not in st.session_state:
        st.session_state[k] = d

DEMO_STEPS = [
    {"label": "Morning \u2014 cool start", "temp": 26, "humidity": 35, "skin_temp": 36.4,
     "moisture": 10, "current_hr": 72, "symptoms": ["None"]},
    {"label": "Midday \u2014 heat building", "temp": 34, "humidity": 45, "skin_temp": 37.0,
     "moisture": 40, "current_hr": 95, "symptoms": ["Mild thirst"]},
    {"label": "Early afternoon \u2014 strain rising", "temp": 39, "humidity": 55, "skin_temp": 37.8,
     "moisture": 65, "current_hr": 118, "symptoms": ["Heavy sweating", "Tiredness / weakness"]},
    {"label": "Peak heat \u2014 danger", "temp": 43, "humidity": 60, "skin_temp": 38.4,
     "moisture": 85, "current_hr": 142, "symptoms": ["Dizziness", "Headache"]},
    {"label": "After rest \u2014 recovering", "temp": 34, "humidity": 45, "skin_temp": 37.0,
     "moisture": 30, "current_hr": 88, "symptoms": ["None"]},
]


def run_check(worker_id, age, weight, height, new_worker, lang, resting_hr,
              temp, humidity, skin_temp, moisture, current_hr, symptoms, log_key="log"):
    wbgt = estimate_wbgt(temp, humidity)
    w_level, w_label = wbgt_risk(wbgt)
    h_level, h_label = hr_risk(resting_hr, current_hr, age)
    s_score = symptom_score(symptoms)
    score = w_level + h_level + (s_score // 2) + (1 if (age >= 50 or new_worker) else 0)

    if s_score >= 5 or score >= 5:
        decision = "\U0001F534 STOP"
    elif score >= 3:
        decision = "\U0001F7E1 REST SOON"
    else:
        decision = "\U0001F7E2 CONTINUE"

    rest_min = rest_minutes(decision, score)
    flag = plausibility_check(wbgt, skin_temp, resting_hr, current_hr)
    ml_prob = ml_risk_probability(wbgt, current_hr, resting_hr, age, moisture)
    risk_percent = ml_prob * 100 if ml_prob is not None else min(100, score / 8 * 100)

    st.session_state[log_key].append({
        "time": datetime.now().strftime("%H:%M:%S"), "worker": worker_id or "unnamed",
        "age": age, "new_worker": new_worker, "temp_c": temp, "humidity_%": humidity,
        "skin_temp_c": skin_temp, "moisture": moisture, "wbgt": wbgt,
        "resting_hr": resting_hr, "current_hr": current_hr,
        "symptoms": ", ".join(symptoms), "symptom_score": s_score,
        "decision": decision, "rest_minutes": rest_min, "flagged": flag is not None,
    })
    return dict(wbgt=wbgt, w_label=w_label, h_label=h_label, s_score=s_score,
                decision=decision, rest_min=rest_min, flag=flag, risk_percent=risk_percent,
                ml_prob=ml_prob, weight=weight, height=height, new_worker=new_worker,
                lang=lang, moisture=moisture, current_hr=current_hr)


def show_result(r, worker_id):
    st.markdown("---")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        render_gauge(r["risk_percent"], key=f"gauge_{len(st.session_state.log)}")
    with c2:
        card("Heart rate", f"{r['current_hr']}", r["h_label"], "\u2764\ufe0f")
        card("Symptom score", r["s_score"], "", "\U0001FA7A")
    with c3:
        card("WBGT estimate", f"{r['wbgt']} \u00b0C", r["w_label"], "\U0001F321\ufe0f")
        card("Moisture", f"{r['moisture']}/100", "", "\U0001F4A7")

    kind = ("danger" if r["decision"] == "\U0001F534 STOP"
            else "warn" if r["decision"] == "\U0001F7E1 REST SOON" else "safe")
    badge(r["decision"], kind)
    if r["ml_prob"] is not None:
        st.caption(f"\U0001F9E0 Trained model estimate: {r['ml_prob']*100:.0f}% probability this worker needs rest.")
    else:
        st.caption("\U0001F9E0 No trained model loaded yet, using rule-based logic.")
    st.caption(bmi_note(r["weight"], r["height"]))
    if r["new_worker"]:
        st.caption("\u26a0\ufe0f New to heat work: thresholds are slightly stricter.")
    if r["flag"]:
        st.warning("\U0001F6A9 " + r["flag"])
    if r["rest_min"] > 0:
        resume = datetime.now() + timedelta(minutes=r["rest_min"])
        st.session_state.rest_until[worker_id or "unnamed"] = resume
        st.warning(f"\u23f1\ufe0f Suggested rest: **{r['rest_min']} minutes** "
                  f"(until ~{resume.strftime('%H:%M')}). Recheck before returning to work.")
    if kind == "danger":
        beep()
        st.error("These readings can indicate heat stroke, a medical emergency. Call for "
                 "medical help immediately, move the worker to a cool place, and begin active "
                 "cooling while waiting for help.")


# ---------------------------------------------------------------------------
# LANDING SCREEN
# ---------------------------------------------------------------------------
if not st.session_state.entered:
    st.markdown("""
    <div class="cg-hero">
      <h1>\U0001F975 CoolGuard</h1>
      <p>Wearable heat stress & dehydration early-warning system for outdoor workers</p>
      <p style="font-size:13px;color:#9aa5ab;">Built for Qatar's outdoor workforce \u2014 Innovators of Tomorrow 2026</p>
    </div>
    """, unsafe_allow_html=True)
    col = st.columns([1, 1, 1])[1]
    with col:
        if st.button("Enter dashboard \u2192", type="primary", use_container_width=True):
            st.session_state.entered = True
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# BAND CONNECTION GATE
# ---------------------------------------------------------------------------
st.markdown("### \U0001F975 CoolGuard")
st.sidebar.header("\U0001F535 Band connection")
if not st.session_state.connected:
    st.sidebar.warning("Not connected")
    if st.sidebar.button("\U0001F50D Scan for band"):
        st.session_state.found_devices = ["CoolGuard-Band-01", "CoolGuard-Band-02"]
    found = st.session_state.get("found_devices", [])
    if found:
        pick = st.sidebar.selectbox("Devices found", found)
        if st.sidebar.button("\U0001F517 Connect"):
            st.session_state.connected = True
            st.session_state.device_name = pick
            st.rerun()
else:
    st.sidebar.success(f"Connected: {st.session_state.device_name}")
    if st.sidebar.button("\U0001F50C Disconnect"):
        st.session_state.connected = False
        st.rerun()

if not st.session_state.connected:
    st.warning("\u26a0\ufe0f No band connected. Scan and connect a device in the sidebar.")
    st.stop()

tab_demo, tab_check, tab_super, tab_log, tab_about = st.tabs(
    ["\u25b6\ufe0f Guided demo", "\u2705 Manual check", "\U0001F9D1\u200d\U0001F4BC Supervisor",
     "\U0001F4CB Log", "\u2139\ufe0f About"]
)

# ---- Guided demo ------------------------------------------------------------
with tab_demo:
    st.caption("A scripted walkthrough for presentations: click Next to move through a worker's "
              "shift and watch the risk rise and fall in real time.")
    step = DEMO_STEPS[st.session_state.demo_step]
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.markdown(f"**Stage {st.session_state.demo_step + 1}/{len(DEMO_STEPS)}:** {step['label']}")
    if c2.button("\u2190 Back", disabled=st.session_state.demo_step == 0):
        st.session_state.demo_step -= 1
        st.rerun()
    if c3.button("Next \u2192", disabled=st.session_state.demo_step == len(DEMO_STEPS) - 1):
        st.session_state.demo_step += 1
        st.rerun()

    r = run_check("DEMO-1", 30, 75.0, 175.0, False, "English", 70,
                  step["temp"], step["humidity"], step["skin_temp"], step["moisture"],
                  step["current_hr"], step["symptoms"], log_key="log")
    show_result(r, "DEMO-1")
    render_trend([row for row in st.session_state.log if row["worker"] == "DEMO-1"],
                key=f"trend_demo_{st.session_state.demo_step}")

# ---- Manual check ------------------------------------------------------------
with tab_check:
    st.subheader("Worker profile")
    c1, c2, c3, c4 = st.columns(4)
    worker_id = c1.text_input("Worker ID / initials", "")
    age = c2.number_input("Age", 16, 75, 30)
    weight = c3.number_input("Weight (kg)", 30.0, 150.0, 75.0, step=1.0)
    height = c4.number_input("Height (cm)", 130.0, 210.0, 170.0, step=1.0)
    c5, c6, c7 = st.columns(3)
    new_worker = c5.checkbox("New to heat work (< 2 weeks)")
    lang = c6.selectbox("Alert language", ["English", "Arabic", "Hindi", "Urdu", "Bengali", "Tagalog"])
    resting_hr = c7.number_input("Resting HR (bpm)", 40, 120, 70)

    st.subheader("Readings")
    h1, h2, h3, h4 = st.columns(4)
    temp = h1.slider("Ambient temp (\u00b0C)", 20.0, 55.0, 38.0, step=0.5)
    humidity = h2.slider("Humidity (%)", 5, 100, 40)
    skin_temp = h3.slider("Skin temp (\u00b0C)", 30.0, 42.0, 36.5, step=0.1)
    moisture = h4.slider("Moisture (0-100)", 0, 100, 30)
    current_hr = st.number_input("Current HR (bpm)", 40, 220, 100)
    symptoms = st.multiselect("Symptoms", list(SYMPTOM_WEIGHTS.keys()), default=["None"])

    if st.button("\U0001F50D Check worker status", type="primary"):
        r = run_check(worker_id, age, weight, height, new_worker, lang, resting_hr,
                      temp, humidity, skin_temp, moisture, current_hr, symptoms)
        show_result(r, worker_id)

    pending = st.session_state.rest_until.get(worker_id)
    if pending:
        remaining = (pending - datetime.now()).total_seconds() / 60
        if remaining > 0:
            st.info(f"\U0001F552 On rest until ~{pending.strftime('%H:%M')} (~{remaining:.0f} min left).")
        else:
            st.success("\u2705 Rest period over. Recheck before returning to work.")

# ---- Supervisor ------------------------------------------------------------
with tab_super:
    if not st.session_state.log:
        st.write("No checks logged yet.")
    else:
        df = pd.DataFrame(st.session_state.log)
        latest = df.sort_values("time").groupby("worker").tail(1)
        c1, c2, c3 = st.columns(3)
        with c1:
            card("STOP", int((latest["decision"] == "\U0001F534 STOP").sum()), "", "\U0001F534")
        with c2:
            card("REST SOON", int((latest["decision"] == "\U0001F7E1 REST SOON").sum()), "", "\U0001F7E1")
        with c3:
            card("CONTINUE", int((latest["decision"] == "\U0001F7E2 CONTINUE").sum()), "", "\U0001F7E2")
        st.dataframe(latest[["worker", "time", "decision", "wbgt", "current_hr",
                             "symptom_score", "rest_minutes", "flagged"]],
                    hide_index=True, use_container_width=True)

# ---- Log ---------------------------------------------------------------
with tab_log:
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("\u2b07\ufe0f Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "coolguard_log.csv", "text/csv")
    else:
        st.write("No checks logged yet.")

# ---- About ---------------------------------------------------------------
with tab_about:
    st.markdown("""
- **WBGT estimate:** simplified shade-only formula from ambient temperature and humidity.
- **Heart rate check:** compares current HR to resting HR and an age-adjusted max.
- **Personalization:** loads a trained model (`coolguard_model.joblib`) if available, otherwise
  falls back to transparent rule-based logic.
- **Plausibility check:** flags readings where the environment and the body's own signals disagree.
- **Guided demo:** a fixed, honest script for live presentations, not live sensor data.
    """)
    st.error("CoolGuard is an educational safety-awareness prototype, not a medical device. "
             "It does not diagnose heat stroke, blood pressure, or blood sugar. In a real "
             "emergency, call for medical help immediately.")
