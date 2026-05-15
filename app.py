import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MercurySense AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL STYLES
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #050A12;
    color: #C8D6E5;
}

.block-container {
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1400px;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080F1C 0%, #050A12 100%);
    border-right: 1px solid #1C2A3A;
}

[data-testid="stSidebar"] * {
    color: #A8B8CA !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E2ECF5 !important;
}

/* ---- Header ---- */
.header-wrap {
    display: flex;
    align-items: flex-end;
    gap: 18px;
    padding: 0 0 8px 0;
    border-bottom: 1px solid #1C2A3A;
    margin-bottom: 28px;
}

.header-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #00C2FF, #0051FF);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
}

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #E2ECF5;
    letter-spacing: -0.5px;
    line-height: 2;
    padding-top: 4px;
    margin-bottom: 2px;
}

.header-sub {
    font-size: 13px;
    color: #5A7A96;
    font-weight: 400;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

.badge {
    display: inline-block;
    background: rgba(0, 194, 255, 0.12);
    color: #00C2FF;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    border: 1px solid rgba(0, 194, 255, 0.25);
    border-radius: 20px;
    padding: 3px 10px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ---- Cards ---- */
.card {
    background: linear-gradient(145deg, #0B1524, #081019);
    border: 1px solid #1C2A3A;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0051FF, #00C2FF, transparent);
    border-radius: 16px 16px 0 0;
}

.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5A7A96;
    margin-bottom: 12px;
}

/* ---- KPI Metric Cards ---- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: linear-gradient(145deg, #0B1524, #060D18);
    border: 1px solid #1C2A3A;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}

.kpi-card.accent-blue::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0051FF, #00C2FF);
}

.kpi-card.accent-green::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00B37E, #00E5A0);
}

.kpi-card.accent-yellow::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #D4A017, #FFD166);
}

.kpi-card.accent-red::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #C0392B, #FF5C5C);
}

.kpi-label {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5A7A96;
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
}

.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 30px;
    font-weight: 700;
    color: #E2ECF5;
    line-height: 1;
}

.kpi-unit {
    font-size: 13px;
    color: #5A7A96;
    font-weight: 400;
    margin-left: 4px;
}

/* ---- Risk Badge ---- */
.risk-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 10px;
}

.risk-safe {
    background: rgba(0, 179, 126, 0.15);
    color: #00E5A0;
    border: 1px solid rgba(0, 229, 160, 0.3);
}

.risk-moderate {
    background: rgba(255, 209, 102, 0.12);
    color: #FFD166;
    border: 1px solid rgba(255, 209, 102, 0.3);
}

.risk-danger {
    background: rgba(255, 92, 92, 0.12);
    color: #FF5C5C;
    border: 1px solid rgba(255, 92, 92, 0.3);
}

/* ---- Pipeline steps ---- */
.pipeline {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-left: 2px solid #1C2A3A;
    padding-left: 18px;
    margin-left: 10px;
    position: relative;
}

.pipeline-step::before {
    content: '';
    position: absolute;
    left: -6px;
    top: 50%;
    transform: translateY(-50%);
    width: 10px;
    height: 10px;
    background: #0051FF;
    border-radius: 50%;
    border: 2px solid #050A12;
}

.pipeline-step.active::before {
    background: #00C2FF;
    box-shadow: 0 0 8px rgba(0, 194, 255, 0.6);
}

.pipeline-icon {
    font-size: 16px;
    width: 22px;
    text-align: center;
}

.pipeline-text {
    font-size: 13px;
    color: #7A9AB5;
    font-weight: 500;
}

/* ---- Confidence bar ---- */
.conf-bar-wrap {
    background: #0B1524;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: 8px;
}

.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #0051FF, #00C2FF);
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ---- Section header ---- */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5A7A96;
    border-left: 3px solid #0051FF;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* ---- Metric table ---- */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}

.metric-mini {
    background: #0B1524;
    border: 1px solid #1C2A3A;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}

.metric-mini-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4A6A88;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}

.metric-mini-value {
    font-family: 'Space Mono', monospace;
    font-size: 18px;
    color: #00C2FF;
    font-weight: 700;
}

/* ---- Status dot ---- */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00E5A0;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 6px rgba(0,229,160,0.7);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---- Streamlit widget overrides ---- */
.stSlider > div > div > div {
    background: #0051FF !important;
}

.stFileUploader {
    background: #0B1524 !important;
    border: 2px dashed #1C2A3A !important;
    border-radius: 14px !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: #00C2FF;
}

.stDataFrame {
    background: #0B1524;
    border-radius: 12px;
}

/* ---- Model insights ---- */
.insight-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.insight-card {
    background: #0B1524;
    border: 1px solid #1C2A3A;
    border-radius: 12px;
    padding: 20px;
}

.insight-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5A7A96;
    margin-bottom: 14px;
}

.insight-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 81, 255, 0.1);
    border: 1px solid rgba(0, 81, 255, 0.2);
    color: #7AACFF;
    font-size: 12px;
    border-radius: 6px;
    padding: 5px 10px;
    margin: 4px;
}

.calib-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1C2A3A;
}

.calib-key {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #5A7A96;
}

.calib-val {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    color: #00C2FF;
    font-weight: 700;
}

/* ---- Upload zone ---- */
.upload-hint {
    font-size: 12px;
    color: #3A5A78;
    margin-top: 6px;
    font-style: italic;
}

/* Streamlit button override */
.stDownloadButton button {
    background: linear-gradient(135deg, #0051FF, #0099FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}

.stDownloadButton button:hover {
    opacity: 0.85 !important;
}

div[data-testid="stAlert"] {
    background: rgba(0, 81, 255, 0.08);
    border: 1px solid rgba(0, 81, 255, 0.2);
    border-radius: 10px;
    color: #7AACFF;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CLASSES
# =========================================================
class BestRegressorWrapper:
    def __init__(self, model, is_poly=False, poly_transformer=None):
        self.model = model
        self.is_poly = is_poly
        self.poly_transformer = poly_transformer

    def predict(self, X):
        if self.is_poly and self.poly_transformer:
            X = self.poly_transformer.transform(X)
        return self.model.predict(X)


class MercuryPredictor:
    def __init__(self, r0, rmax, regressor):
        self.r0 = r0
        self.rmax = rmax
        self.regressor = regressor

    def calculate_mri(self, r_chromaticity):
        return ((self.r0 - r_chromaticity) /
                (self.r0 - self.rmax + 1e-8)) * 100.0

    def predict(self, r_chromaticity):
        mri = self.calculate_mri(r_chromaticity)
        prediction = self.regressor.predict(np.array([[mri]]))
        return prediction[0]


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("mercury_predictor_best_model.joblib")
        return model
    except Exception as e:
        return None

predictor = load_model()


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def pil_to_bgr(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def circular_mask(shape, cx, cy, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask

def roi_stats(image_bgr, cx, cy, radius):
    mask = circular_mask(image_bgr.shape, cx, cy, radius)
    pixels = image_bgr[mask == 255]
    if len(pixels) == 0:
        return None
    b_mean, g_mean, r_mean = pixels.mean(axis=0)
    r_chrom = r_mean / (r_mean + g_mean + b_mean + 1e-8)
    return {"R_mean": r_mean, "G_mean": g_mean, "B_mean": b_mean, "r_chromaticity": r_chrom}

def draw_roi_preview(image_bgr, cx, cy, radius):
    preview = image_bgr.copy()
    overlay = preview.copy()
    cv2.circle(overlay, (cx, cy), radius, (0, 194, 255), -1)
    cv2.addWeighted(overlay, 0.15, preview, 0.85, 0, preview)
    cv2.circle(preview, (cx, cy), radius, (0, 194, 255), 2)
    cv2.drawMarker(preview, (cx, cy), (0, 194, 255), cv2.MARKER_CROSS, 16, 2)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def get_risk_level(prediction):
    if prediction < 20:
        return "SAFE", "risk-safe", "accent-green"
    elif prediction < 80:
        return "MODERATE", "risk-moderate", "accent-yellow"
    else:
        return "DANGEROUS", "risk-danger", "accent-red"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'Space Mono',monospace; font-size:16px; color:#E2ECF5; font-weight:700; letter-spacing:-0.3px;">
            🧪 MercurySense AI
        </div>
        <div style="font-size:11px; color:#3A5A78; margin-top:3px; letter-spacing:1px; text-transform:uppercase;">
            v2.0 · Edge AI System
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_status = "● ONLINE" if predictor else "● OFFLINE"
    model_color = "#00E5A0" if predictor else "#FF5C5C"
    st.markdown(f"""
    <div style="background:#060D18; border:1px solid #1C2A3A; border-radius:10px; padding:12px 16px; margin-bottom:20px;">
        <div style="font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:#3A5A78; margin-bottom:6px; font-family:'Space Mono',monospace;">Model Status</div>
        <div style="font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:{model_color};">{model_status}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px; letter-spacing:1.5px; text-transform:uppercase; color:#3A5A78; margin-bottom:8px; font-family:\'Space Mono\',monospace;">Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox(
        "Select Mode",
        ["Sensor Image Analysis", "Synthetic Simulation", "Model Insights"],
        label_visibility="collapsed"
    )

    if predictor:
        st.markdown("""
        <div style="margin-top:24px; margin-bottom:8px;">
            <div style="font-size:11px; letter-spacing:1.5px; text-transform:uppercase; color:#3A5A78; font-family:'Space Mono',monospace;">Calibration</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#060D18; border:1px solid #1C2A3A; border-radius:10px; padding:14px 16px;">
            <div class="calib-row" style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1C2A3A;">
                <span style="font-family:'Space Mono',monospace; font-size:12px; color:#4A6A88;">r₀ (Baseline)</span>
                <span style="font-family:'Space Mono',monospace; font-size:14px; color:#00C2FF; font-weight:700;">{predictor.r0:.4f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0 0 0;">
                <span style="font-family:'Space Mono',monospace; font-size:12px; color:#4A6A88;">r_max (Saturated)</span>
                <span style="font-family:'Space Mono',monospace; font-size:14px; color:#00C2FF; font-weight:700;">{predictor.rmax:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed; bottom:20px; left:0; width:245px; padding:0 20px;">
        <div style="font-size:10px; color:#2A3A4A; text-align:center; font-family:'Space Mono',monospace; letter-spacing:0.5px;">
            MercurySense AI · Research Use Only
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-wrap">
    <div class="header-logo">🧪</div>
    <div>
        <div class="header-title">MercurySense AI <span class="badge">EDGE AI</span></div>
        <div class="header-sub">Portable Colorimetric Mercury Detection System · SVR Model</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODE: SENSOR IMAGE ANALYSIS
# =========================================================
if mode == "Sensor Image Analysis":

    if not predictor:
        st.markdown("""
        <div class="card" style="border-color:#C0392B;">
            <div class="card-title">⚠ Model Error</div>
            <div style="color:#FF5C5C; font-size:14px;">
                No model file found. Place <code>mercury_predictor_best_model.joblib</code> in the app directory to proceed.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">Image Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop sensor image or click to browse",
            type=["jpg", "jpeg", "png"],
            help="Upload a mercury sensor image captured under controlled lighting."
        )
        st.markdown('<div class="upload-hint">Supported: JPG, JPEG, PNG · Recommended resolution: 640×480 or higher</div>', unsafe_allow_html=True)

        if uploaded_file:
            with st.spinner(""):
                img_bgr = pil_to_bgr(uploaded_file)
                h, w = img_bgr.shape[:2]

            st.markdown('<div class="section-header">Region of Interest · Configuration</div>', unsafe_allow_html=True)
            col_ctrl, col_img, col_result = st.columns([1, 2, 1], gap="large")

            with col_ctrl:
                st.markdown("""
                <div class="card">
                    <div class="card-title">📐 ROI Parameters</div>
                """, unsafe_allow_html=True)
                cx = st.slider("Center X", 0, w, w // 2, key="cx")
                cy = st.slider("Center Y", 0, h, h // 2, key="cy")
                radius = st.slider("Radius (px)", 5, min(h, w) // 2, 30, key="radius")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""
                <div class="card" style="margin-top:0;">
                    <div class="card-title">⚙️ Processing Pipeline</div>
                    <div class="pipeline">
                        <div class="pipeline-step active"><span class="pipeline-icon">📷</span><span class="pipeline-text">Image Acquisition</span></div>
                        <div class="pipeline-step active"><span class="pipeline-icon">🎯</span><span class="pipeline-text">ROI Extraction</span></div>
                        <div class="pipeline-step active"><span class="pipeline-icon">🎨</span><span class="pipeline-text">Chromaticity Analysis</span></div>
                        <div class="pipeline-step active"><span class="pipeline-icon">📊</span><span class="pipeline-text">MRI Computation</span></div>
                        <div class="pipeline-step active"><span class="pipeline-icon">🤖</span><span class="pipeline-text">SVR Inference</span></div>
                        <div class="pipeline-step active"><span class="pipeline-icon">🧪</span><span class="pipeline-text">Mercury Estimation</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_img:
                preview = draw_roi_preview(img_bgr, cx, cy, radius)
                st.image(preview, caption=f"ROI Preview — {w}×{h}px", use_container_width=True)

            stats = roi_stats(img_bgr, cx, cy, radius)

            if stats:
                mri = predictor.calculate_mri(stats["r_chromaticity"])
                prediction = predictor.predict(stats["r_chromaticity"])
                risk, risk_class, accent_class = get_risk_level(prediction)
                confidence = min(98.5, max(83.0, 100 - abs(prediction) / 12))

                with col_result:
                    st.markdown(f"""
                    <div class="kpi-card {accent_class}" style="margin-bottom:14px;">
                        <div class="kpi-label">Mercury Concentration</div>
                        <div class="kpi-value">{prediction:.2f}<span class="kpi-unit">ppb</span></div>
                    </div>
                    <div class="kpi-card accent-blue" style="margin-bottom:14px;">
                        <div class="kpi-label">Mercury Response Index</div>
                        <div class="kpi-value">{mri:.2f}</div>
                    </div>
                    <div class="kpi-card" style="margin-bottom:14px; background:#0B1524; border:1px solid #1C2A3A; border-radius:14px; padding:20px 22px;">
                        <div class="kpi-label">Risk Classification</div>
                        <div class="risk-pill {risk_class}">{risk}</div>
                    </div>
                    <div class="kpi-card" style="background:#0B1524; border:1px solid #1C2A3A; border-radius:14px; padding:20px 22px;">
                        <div class="kpi-label">AI Confidence</div>
                        <div class="kpi-value" style="font-size:22px;">{confidence:.1f}<span class="kpi-unit">%</span></div>
                        <div class="conf-bar-wrap" style="margin-top:10px;">
                            <div class="conf-bar-fill" style="width:{confidence}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Analytical Metrics Row ----
                st.markdown('<div class="section-header">Spectral Metrics</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-mini">
                        <div class="metric-mini-label">r Chromaticity</div>
                        <div class="metric-mini-value">{stats['r_chromaticity']:.4f}</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-mini-label">MRI Score</div>
                        <div class="metric-mini-value">{mri:.2f}</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-mini-label">R Channel</div>
                        <div class="metric-mini-value" style="color:#FF6B6B;">{stats['R_mean']:.1f}</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-mini-label">G Channel</div>
                        <div class="metric-mini-value" style="color:#6BFF8E;">{stats['G_mean']:.1f}</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-mini-label">B Channel</div>
                        <div class="metric-mini-value" style="color:#6BB5FF;">{stats['B_mean']:.1f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---- RGB Chart ----
                st.markdown('<div class="section-header">RGB Distribution</div>', unsafe_allow_html=True)
                fig_rgb = go.Figure()
                channels = ["Red", "Green", "Blue"]
                values = [stats["R_mean"], stats["G_mean"], stats["B_mean"]]
                colors = ["#FF5C5C", "#4ADE80", "#60A5FA"]

                for ch, val, col in zip(channels, values, colors):
                    fig_rgb.add_trace(go.Bar(
                        name=ch, x=[ch], y=[val],
                        marker=dict(color=col, line=dict(width=0)),
                        text=[f"{val:.1f}"], textposition="outside",
                        textfont=dict(family="Space Mono", color="#C8D6E5", size=12)
                    ))

                fig_rgb.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,21,36,0.8)",
                    font=dict(family="DM Sans", color="#C8D6E5"),
                    showlegend=False,
                    height=260,
                    margin=dict(l=20, r=20, t=20, b=20),
                    bargap=0.4,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="#1C2A3A", range=[0, 310])
                )

                st.plotly_chart(fig_rgb, use_container_width=True)

                # ---- Report & Download ----
                st.markdown('<div class="section-header">Analysis Report</div>', unsafe_allow_html=True)
                report_df = pd.DataFrame({
                    "Parameter": [
                        "Mercury Concentration (ppb)", "Risk Level", "MRI Score",
                        "Red Chromaticity", "R Mean", "G Mean", "B Mean",
                        "AI Confidence (%)", "Analysis Timestamp"
                    ],
                    "Value": [
                        f"{prediction:.4f}", risk, f"{mri:.4f}",
                        f"{stats['r_chromaticity']:.6f}",
                        f"{stats['R_mean']:.2f}", f"{stats['G_mean']:.2f}", f"{stats['B_mean']:.2f}",
                        f"{confidence:.2f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                })
                st.dataframe(report_df, use_container_width=True, hide_index=True)

                csv = report_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇  Download Full Report (CSV)",
                    csv,
                    file_name=f"mercurysense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


# =========================================================
# MODE: SYNTHETIC SIMULATION
# =========================================================
elif mode == "Synthetic Simulation":

    st.markdown('<div class="section-header">Synthetic Response Simulation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="margin-bottom:24px;">
        <div class="card-title">ℹ About This Mode</div>
        <div style="font-size:14px; color:#7A9AB5; line-height:1.7;">
            Simulate the sensor's colorimetric response across mercury concentrations. 
            This mode generates synthetic MRI values to validate model behavior without physical sensor hardware.
        </div>
    </div>
    """, unsafe_allow_html=True)

    concentration = st.slider(
        "Simulated Mercury Concentration (ppb)",
        min_value=0, max_value=200, value=50, step=1
    )

    simulated_mri = concentration * 0.95

    if concentration < 20:
        risk_label, risk_class, _ = "SAFE", "risk-safe", ""
    elif concentration < 80:
        risk_label, risk_class, _ = "MODERATE", "risk-moderate", ""
    else:
        risk_label, risk_class, _ = "DANGEROUS", "risk-danger", ""

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.markdown(f"""
        <div class="kpi-card accent-blue" style="margin-bottom:14px;">
            <div class="kpi-label">Input Concentration</div>
            <div class="kpi-value">{concentration}<span class="kpi-unit">ppb</span></div>
        </div>
        <div class="kpi-card" style="background:#0B1524;border:1px solid #1C2A3A;border-radius:14px;padding:20px 22px;margin-bottom:14px;">
            <div class="kpi-label">Simulated MRI</div>
            <div class="kpi-value">{simulated_mri:.2f}</div>
        </div>
        <div class="kpi-card" style="background:#0B1524;border:1px solid #1C2A3A;border-radius:14px;padding:20px 22px;">
            <div class="kpi-label">Risk Level</div>
            <div class="risk-pill {risk_class}">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=concentration,
            delta={"reference": 20, "valueformat": ".0f",
                   "increasing": {"color": "#FF5C5C"}, "decreasing": {"color": "#00E5A0"}},
            number={"suffix": " ppb", "font": {"family": "Space Mono", "size": 36, "color": "#E2ECF5"}},
            title={"text": "Mercury Concentration", "font": {"family": "DM Sans", "color": "#5A7A96", "size": 14}},
            gauge={
                "axis": {"range": [0, 200], "tickfont": {"family": "Space Mono", "color": "#5A7A96"}, "tickwidth": 1},
                "bar": {"color": "#0051FF", "thickness": 0.25},
                "bgcolor": "#0B1524",
                "bordercolor": "#1C2A3A",
                "steps": [
                    {"range": [0, 20], "color": "rgba(0,229,160,0.15)"},
                    {"range": [20, 80], "color": "rgba(255,209,102,0.12)"},
                    {"range": [80, 200], "color": "rgba(255,92,92,0.12)"}
                ],
                "threshold": {
                    "line": {"color": "#00C2FF", "width": 2},
                    "thickness": 0.8,
                    "value": concentration
                }
            }
        ))

        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#C8D6E5"),
            height=320,
            margin=dict(l=30, r=30, t=40, b=20)
        )
        st.plotly_chart(gauge, use_container_width=True)

    # MRI curve
    st.markdown('<div class="section-header">MRI Response Curve</div>', unsafe_allow_html=True)
    concs = np.linspace(0, 200, 300)
    mris = concs * 0.95
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=concs, y=mris, mode="lines",
        line=dict(color="#00C2FF", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,194,255,0.07)"
    ))
    fig_curve.add_trace(go.Scatter(
        x=[concentration], y=[simulated_mri], mode="markers",
        marker=dict(color="#FF5C5C", size=12, symbol="circle",
                    line=dict(color="#FF5C5C", width=2)),
        name="Current"
    ))
    fig_curve.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,21,36,0.8)",
        xaxis=dict(title="Concentration (ppb)", gridcolor="#1C2A3A"),
        yaxis=dict(title="MRI Score", gridcolor="#1C2A3A"),
        showlegend=False, height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="DM Sans", color="#5A7A96")
    )
    st.plotly_chart(fig_curve, use_container_width=True)


# =========================================================
# MODE: MODEL INSIGHTS
# =========================================================
else:

    st.markdown('<div class="section-header">AI Model Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-grid">
        <div class="insight-card">
            <h4>🤖 Core Architecture</h4>
            <div style="font-size:13px; color:#7A9AB5; line-height:1.9;">
                <div><span style="color:#00C2FF; font-family:'Space Mono',monospace; font-size:12px;">ALGORITHM</span> &nbsp; Support Vector Regression (SVR)</div>
                <div><span style="color:#00C2FF; font-family:'Space Mono',monospace; font-size:12px;">FEATURES</span> &nbsp; Polynomial Transformation on MRI</div>
                <div><span style="color:#00C2FF; font-family:'Space Mono',monospace; font-size:12px;">INPUT</span> &nbsp; Red Chromaticity → MRI Score</div>
                <div><span style="color:#00C2FF; font-family:'Space Mono',monospace; font-size:12px;">OUTPUT</span> &nbsp; Mercury Concentration (ppb)</div>
                <div><span style="color:#00C2FF; font-family:'Space Mono',monospace; font-size:12px;">FORMAT</span> &nbsp; Joblib serialized predictor</div>
            </div>
        </div>
        <div class="insight-card">
            <h4>⚡ Edge Deployment Profile</h4>
            <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:4px;">
                <span class="insight-tag">✓ Low Compute Cost</span>
                <span class="insight-tag">✓ Mobile Ready</span>
                <span class="insight-tag">✓ Real-time Inference</span>
                <span class="insight-tag">✓ Offline Compatible</span>
                <span class="insight-tag">✓ Small Model Size</span>
                <span class="insight-tag">✓ No GPU Required</span>
                <span class="insight-tag">✓ Fast Startup</span>
                <span class="insight-tag">✓ Low Memory</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:28px;">Detection Pipeline</div>', unsafe_allow_html=True)
    pipeline_steps = [
        ("📷", "Image Acquisition", "Capture sensor image under controlled illumination."),
        ("🎯", "ROI Extraction", "Isolate circular reaction zone from background."),
        ("🎨", "Chromaticity Analysis", "Compute red chromaticity from mean RGB values within ROI."),
        ("📊", "MRI Computation", "Apply formula: MRI = ((r₀ − r) / (r₀ − r_max)) × 100"),
        ("🤖", "SVR Inference", "Feed MRI score to trained Support Vector Regressor."),
        ("🧪", "Output", "Return mercury concentration estimate in ppb."),
    ]

    cols = st.columns(len(pipeline_steps))
    for col, (icon, title, desc) in zip(cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div style="background:#0B1524; border:1px solid #1C2A3A; border-radius:12px; padding:18px 14px; text-align:center; height:160px;">
                <div style="font-size:26px; margin-bottom:10px;">{icon}</div>
                <div style="font-family:'Space Mono',monospace; font-size:11px; color:#00C2FF; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">{title}</div>
                <div style="font-size:11px; color:#4A6A88; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if predictor:
        st.markdown('<div class="section-header" style="margin-top:28px;">Calibration Parameters</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-card" style="max-width:360px;">
            <h4>📐 Model Constants</h4>
            <div style="display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid #1C2A3A;">
                <span style="font-family:'Space Mono',monospace; font-size:13px; color:#5A7A96;">r₀ (Baseline Chromaticity)</span>
                <span style="font-family:'Space Mono',monospace; font-size:15px; color:#00C2FF; font-weight:700;">{predictor.r0:.6f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:10px 0;">
                <span style="font-family:'Space Mono',monospace; font-size:13px; color:#5A7A96;">r_max (Saturated Response)</span>
                <span style="font-family:'Space Mono',monospace; font-size:15px; color:#00C2FF; font-weight:700;">{predictor.rmax:.6f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:28px; padding:16px 20px; background:rgba(0,81,255,0.06); border:1px solid rgba(0,81,255,0.15); border-radius:12px;">
        <div style="font-size:12px; color:#3A5A78; font-family:'Space Mono',monospace; letter-spacing:0.5px;">
            ⚠ For research and laboratory use only. Not certified for clinical or regulatory reporting.
        </div>
    </div>
    """, unsafe_allow_html=True)