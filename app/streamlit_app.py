"""Streamlit interface for the osteoporosis prediction API."""
from __future__ import annotations

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import API_BASE_URL as CONFIG_API_BASE_URL

API_BASE_URL = CONFIG_API_BASE_URL.rstrip("/")

st.set_page_config(
    page_title="Osteoporosis Risk Lab",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #17252b; --muted: #52656b; --mint: #d9eee5; --coral: #e77c67; }
    .stApp { background: #f7f4ee; color: var(--ink); }
    [data-testid="stSidebar"] { background: #17343a; }
    [data-testid="stSidebar"] * { color: #edf7f1; }
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] [data-testid="InputInstructions"],
    [data-testid="stAppViewContainer"] [data-testid="InputInstructions"] p {
        color: var(--ink) !important;
        font-weight: 700 !important;
    }
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea,
    [data-testid="stAppViewContainer"] select,
    [data-testid="stAppViewContainer"] [data-baseweb="select"],
    [data-testid="stAppViewContainer"] [data-baseweb="select"] *,
    [data-testid="stAppViewContainer"] [data-testid="stBaseInput"] *,
    [data-testid="stAppViewContainer"] .stNumberInput input,
    [data-testid="stAppViewContainer"] .stTextInput input,
    [data-testid="stAppViewContainer"] .stTextArea textarea,
    [data-testid="stAppViewContainer"] .stSelectbox div[role="combobox"],
    [data-testid="stAppViewContainer"] .stSlider [role="slider"] {
        color: #111827 !important;
        background: #ffffff !important;
        font-weight: 700 !important;
        border-color: #cbd5e1 !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput input,
    [data-testid="stAppViewContainer"] .stNumberInput input,
    [data-testid="stAppViewContainer"] .stTextArea textarea,
    [data-testid="stAppViewContainer"] .stSelectbox div[role="combobox"],
    [data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        min-height: 2.75rem !important;
        height: auto !important;
        padding: 0.55rem 0.75rem !important;
        line-height: 1.4 !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    [data-testid="stAppViewContainer"] .stSlider [role="slider"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    .field-help {
        color: var(--muted) !important;
        font-size: 0.82rem;
        line-height: 1.35;
        margin: -0.45rem 0 0.65rem;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    .eyebrow { color: #317c6a; font-size: .78rem; font-weight: 700; letter-spacing: .08rem; text-transform: uppercase; }
    .notice { border-left: 4px solid var(--coral); background: #fff1eb; padding: .8rem 1rem; color: #5b3027; }
    .result-panel { background: var(--mint); padding: 1.2rem 1.4rem; border-radius: 8px; }
    .result-metric { margin: 0 0 1rem; }
    .metric-label { color: var(--ink); font-size: .9rem; font-weight: 700; margin-bottom: .2rem; }
    .metric-value { color: var(--ink); font-size: 1.8rem; font-weight: 800; line-height: 1.15; }
    .risk-normal { color: #15803d; }
    .risk-osteopenia { color: #c2410c; }
    .risk-osteoporosis { color: #b91c1c; }
    [data-testid="stDataFrame"] *, [data-testid="stTable"] * { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params):
    try:
        response = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=8)
        response.raise_for_status()
        return _response_json(response), None
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, payload: dict):
    try:
        response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return _response_json(response), None
    except httpx.HTTPStatusError as exc:
        body = _response_json(exc.response)
        detail = body.get("detail") if isinstance(body, dict) else None
        detail = detail or exc.response.text.strip() or "The API returned an empty response."
        return None, str(detail)
    except Exception as exc:
        return None, str(exc)


def _response_json(response: httpx.Response) -> dict | list:
    """Decode an API response without crashing on empty or invalid JSON."""
    try:
        payload = response.json()
    except ValueError:
        return {"detail": response.text.strip() or "The API returned an empty response."}
    return payload if isinstance(payload, (dict, list)) else {"detail": "The API returned an invalid response."}


def risk_gauge(probabilities: dict[str, float], risk_label: str):
    labels = list(probabilities)
    values = [probabilities[label] * 100 for label in labels]
    colors = ["#39a879", "#e3a63d", "#d65d4b"]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors[:len(labels)]))
    figure.update_layout(
        title=f"Predicted profile: {risk_label}",
        xaxis_title="Probability (%)",
        xaxis_range=[0, 100],
        height=260,
        margin=dict(l=10, r=10, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17252b"),
        title_font=dict(color="#17252b"),
    )
    figure.update_xaxes(title_font=dict(color="#17252b"), tickfont=dict(color="#17252b"))
    figure.update_yaxes(title_font=dict(color="#17252b"), tickfont=dict(color="#17252b"))
    return figure


def select_with_help(label: str, options: list[str], help_text: str) -> str:
    """Render a selectbox with both a hover tooltip and visible guidance."""
    value = st.selectbox(label, options, help=help_text)
    st.markdown(f'<div class="field-help">{help_text}</div>', unsafe_allow_html=True)
    return value


def patient_form() -> dict:
    with st.form("patient_form"):
        st.markdown('<div class="eyebrow">Patient profile</div>', unsafe_allow_html=True)
        st.subheader("Core measurements")
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        gender = select_with_help("Gender", ["Female", "Male", "Other"], "Select the patient's reported gender.")
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=68.0, step=0.5)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.metric("Calculated BMI", f"{bmi:.1f}")
        race = select_with_help(
            "Race / ethnicity",
            ["Caucasian", "African American", "Hispanic", "Asian", "Other"],
            "Select the patient's reported race or ethnicity.",
        )
        st.subheader("Lifestyle")
        calcium = select_with_help(
            "Calcium intake", ["Low", "Adequate", "High"],
            "Low: below usual needs; Adequate: meets usual needs; High: above usual needs.",
        )
        vitamin_d = select_with_help(
            "Vitamin D intake", ["Deficient", "Sufficient", "High"],
            "Deficient: low intake; Sufficient: meets usual needs; High: above usual needs.",
        )
        activity = select_with_help(
            "Physical activity", ["Sedentary", "Active", "Very Active"],
            "Sedentary: little exercise; Active: regular exercise; Very Active: frequent exercise.",
        )
        smoking = select_with_help("Smoking", ["No", "Yes"], "Whether the patient currently smokes.")
        alcohol = select_with_help(
            "Alcohol consumption", ["None", "Moderate", "High"],
            "None: no alcohol; Moderate: occasional use; High: frequent or heavy use.",
        )
        st.subheader("Medical history")
        family = select_with_help(
            "Family history", ["No", "Yes"],
            "Whether a close family member has had osteoporosis or a fragility fracture.",
        )
        fractures = select_with_help(
            "Prior fractures", ["No", "Yes"],
            "Whether the patient has previously had a bone fracture.",
        )
        hormonal = select_with_help(
            "Hormonal changes", ["Normal", "Postmenopausal", "Deficiency"],
            "Select the patient's current hormonal status or known deficiency.",
        )
        conditions = st.text_input("Medical conditions", value="None")
        medications = st.text_input("Medications", value="None")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted:
        return {}
    payload = {
        "Age": age, "Gender": gender, "Height": height_cm, "Weight": weight_kg,
        "BMI": bmi, "Race/Ethnicity": race, "Calcium Intake": calcium,
        "Vitamin D Intake": vitamin_d, "Physical Activity": activity,
        "Smoking": smoking, "Alcohol Consumption": alcohol,
        "Family History": family, "Prior Fractures": fractures,
        "Hormonal Changes": hormonal, "Medical Conditions": conditions,
        "Medications": medications,
    }
    return payload


def render_history() -> None:
    st.subheader("Prediction history")
    history, history_error = api_get("/history", limit=50)
    if history_error:
        st.error(f"History unavailable: {history_error}")
        return
    if not isinstance(history, list) or not history:
        st.caption("No history yet")
        return
    rows = []
    for item in history:
        input_data = item.get("input_data") or {}
        scores = item.get("confidence_scores") or item.get("probabilities") or {}
        rows.append({
            "Timestamp": item.get("created_at"),
            "Age": input_data.get("Age"),
            "Gender": input_data.get("Gender"),
            "Predicted risk": item.get("prediction_result") or item.get("risk_label"),
            "Confidence": ", ".join(f"{label}: {float(value):.1%}" for label, value in scores.items()),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


st.title("Osteoporosis Risk Lab")
st.markdown("Estimate a three-class risk profile from patient-level clinical and lifestyle data.")
st.markdown(
    '<div class="notice">Research support only. This model does not diagnose osteoporosis. Discuss results with a qualified clinician.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    page = st.radio("View", ["Predict", "History"])
    st.divider()
    status, status_error = api_get("/health")

form_payload = patient_form() if page == "Predict" else {}
if form_payload:
    result, error = api_post("/predict", form_payload)
    if error:
        st.error(f"Prediction unavailable: {error}")
    else:
        left, right = st.columns([1.1, 1.4])
        with left:
            risk_class = result["risk_label"].lower().replace(" ", "-")
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-metric"><div class="metric-label">Risk classification</div>'
                f'<div class="metric-value risk-{risk_class}">{result["risk_label"]}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="result-metric"><div class="metric-label">Model confidence</div>'
                f'<div class="metric-value">{result["confidence"] * 100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.write(result["explanation"])
        with right:
            st.plotly_chart(risk_gauge(result["probabilities"], result["risk_label"]), use_container_width=True)

        st.subheader("Contributing factors")
        factors = pd.DataFrame(result.get("contributing_factors", []))
        if factors.empty:
            st.info("No SHAP factors were returned. The prediction remains available without optional explainability dependencies.")
        else:
            factors["direction"] = factors["impact"].map(lambda value: "Higher risk" if value > 0 else "Lower risk")
            st.bar_chart(factors.set_index("feature")["impact"])
            st.dataframe(factors[["feature", "impact", "direction"]], hide_index=True, use_container_width=True)

if page == "History":
    render_history()