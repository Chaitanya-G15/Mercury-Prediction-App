import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import io
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Mercury Colorimetric Analyzer", layout="wide")

st.title("Mercury Colorimetric Analyzer")
st.caption("Classical image-analysis workflow for mercury paper sensor images — designed to run locally with Streamlit in VS Code.")

DEFAULT_R0 = 0.72
DEFAULT_RMAX = 0.18
DEFAULT_SLOPE = 200.0
DEFAULT_INTERCEPT = 0.0


def pil_to_bgr(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return bgr


def circular_mask(shape, cx, cy, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def roi_stats(image_bgr, cx, cy, radius):
    mask = circular_mask(image_bgr.shape, cx, cy, radius)
    pixels = image_bgr[mask == 255]
    b_mean, g_mean, r_mean = pixels.mean(axis=0)
    red_chromaticity = r_mean / (r_mean + g_mean + b_mean + 1e-8)
    return {
        "R_mean": float(r_mean),
        "G_mean": float(g_mean),
        "B_mean": float(b_mean),
        "r_chromaticity": float(red_chromaticity),
        "mask": mask,
    }


def mercury_response_index(r_value, r0, rmax):
    return ((r0 - r_value) / (r0 - rmax + 1e-8)) * 100.0


def predict_concentration(mri, slope, intercept):
    return slope * (mri / 100.0) + intercept


def draw_roi_preview(image_bgr, cx, cy, radius):
    preview = image_bgr.copy()
    cv2.circle(preview, (cx, cy), radius, (0, 255, 255), 2)
    cv2.circle(preview, (cx, cy), 3, (0, 255, 255), -1)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def generate_realistic_sample(conc, size=256, cmax=200):
    img = np.ones((size, size, 3), dtype=np.uint8) * np.random.randint(220, 245)
    noise = np.random.normal(0, 4, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cx = size // 2 + np.random.randint(-10, 11)
    cy = size // 2 + np.random.randint(-10, 11)
    radius = np.random.randint(42, 56)

    red = int(np.clip(220 - (conc / cmax) * 170 + np.random.randint(-8, 9), 40, 220))
    green = np.random.randint(8, 28)
    blue = np.random.randint(8, 22)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    overlay = img.copy()
    overlay[mask == 255] = (blue, green, red)
    alpha = np.random.uniform(0.82, 0.95)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    x = np.linspace(0.85, 1.15, size)
    y = np.linspace(0.90, 1.10, size)
    xv, yv = np.meshgrid(x, y)
    grad = (xv * yv).astype(np.float32)
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= grad
    img = np.clip(out, 0, 255).astype(np.uint8)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb, cx, cy, radius


with st.sidebar:
    st.header("Calibration")
    r0 = st.number_input("Baseline r0 (0 ppb)", value=DEFAULT_R0, format="%.4f")
    rmax = st.number_input("High concentration rmax", value=DEFAULT_RMAX, format="%.4f")
    slope = st.number_input("Regression slope", value=DEFAULT_SLOPE, format="%.4f")
    intercept = st.number_input("Regression intercept", value=DEFAULT_INTERCEPT, format="%.4f")
    st.markdown("Use your experimentally derived values here after calibration.")

mode = st.radio("Choose mode", ["Analyze uploaded image", "Generate realistic demo image"], horizontal=True)

if mode == "Analyze uploaded image":
    uploaded = st.file_uploader("Upload a sensor image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image_bgr = pil_to_bgr(uploaded)
        h, w = image_bgr.shape[:2]

        st.subheader("ROI selection")
        c1, c2, c3 = st.columns(3)
        with c1:
            cx = st.slider("Center X", 0, w - 1, w // 2)
        with c2:
            cy = st.slider("Center Y", 0, h - 1, h // 2)
        with c3:
            radius = st.slider("Radius", 5, min(h, w) // 2, min(h, w) // 6)

        stats = roi_stats(image_bgr, cx, cy, radius)
        mri = mercury_response_index(stats["r_chromaticity"], r0, rmax)
        pred = predict_concentration(mri, slope, intercept)

        left, right = st.columns([1.2, 1])
        with left:
            st.image(draw_roi_preview(image_bgr, cx, cy, radius), caption="ROI preview", use_container_width=True)
        with right:
            metrics = pd.DataFrame({
                "Metric": ["R mean", "G mean", "B mean", "Red chromaticity r", "Mercury Response Index", "Predicted Hg concentration"],
                "Value": [
                    round(stats["R_mean"], 3),
                    round(stats["G_mean"], 3),
                    round(stats["B_mean"], 3),
                    round(stats["r_chromaticity"], 5),
                    round(mri, 3),
                    round(pred, 3),
                ]
            })
            st.dataframe(metrics, use_container_width=True, hide_index=True)

        csv = metrics.to_csv(index=False).encode("utf-8")
        st.download_button("Download analysis CSV", data=csv, file_name="mercury_analysis.csv", mime="text/csv")
    else:
        st.info("Upload an image to begin analysis.")

else:
    conc = st.slider("Simulated mercury concentration (ppb)", 0, 200, 50, 5)
    rgb, cx, cy, radius = generate_realistic_sample(conc)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    stats = roi_stats(bgr, cx, cy, radius)
    mri = mercury_response_index(stats["r_chromaticity"], r0, rmax)
    pred = predict_concentration(mri, slope, intercept)

    left, right = st.columns([1.2, 1])
    with left:
        preview = draw_roi_preview(bgr, cx, cy, radius)
        st.image(preview, caption=f"Generated realistic sample at {conc} ppb", use_container_width=True)
    with right:
        st.write({
            "center_x": cx,
            "center_y": cy,
            "radius": radius,
            "red_chromaticity": round(stats["r_chromaticity"], 5),
            "mercury_response_index": round(mri, 3),
            "predicted_concentration_ppb": round(pred, 3),
        })

st.markdown("---")

