import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta, timezone
import csv, os

ROOT = Path.cwd()
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = ROOT / "plots"; PLOTS_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="Forest Fire Risk Predictor", page_icon="🔥", layout="centered")
st.title("🔥 Forest Fire Risk Prediction Dashboard (Optimized Ensemble)")

# Sidebar - Settings
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("### Ensemble Weights (Optimized Defaults)")
w_rf = st.sidebar.slider("RF Weight", 0.0, 1.0, 1.0, 0.05)
w_cnn = st.sidebar.slider("CNN Weight", 0.0, 1.0, 0.3, 0.05)
w_lstm = st.sidebar.slider("LSTM Weight", 0.0, 1.0, 0.3, 0.05)
thr = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.40, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Available Cities:**")
cities = {
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "elev": 450},
    "Almora": {"lat": 29.5973, "lon": 79.6591, "elev": 1600},
    "Haridwar": {"lat": 29.9457, "lon": 78.1642, "elev": 330},
}
city = st.sidebar.selectbox("Choose location", list(cities.keys()))
city_coords = (cities[city]["lat"], cities[city]["lon"])
default_elev = cities[city].get("elev", 450)

st.sidebar.markdown("---")
st.sidebar.markdown("📡 Live weather")
use_live = st.sidebar.checkbox("Use Open-Meteo live 7-day data", value=False)
mode = st.sidebar.radio("Mode", ("Past 7 days (observed)", "Next 7 days (forecast)"))
st.sidebar.caption("Past: aggregates last 7 days' hourly observations. Next: fetches forecast for the coming 7 days.")

#Model Loading
@st.cache_resource(show_spinner=False)
def load_rf(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource(show_spinner=False)
def load_keras(path: Path):
    if not path.exists():
        return None
    try:
        return load_model(str(path), compile=False)
    except Exception as e:
        st.sidebar.warning(f"Could not load {path.name}: {e}")
        return None

rf = load_rf(MODELS_DIR / "randomforest.pkl")
cnn = load_keras(MODELS_DIR / "cnn_tuned_model_final.h5")
lstm = load_keras(MODELS_DIR / "lstm_tuned_model_final.h5")

if rf is None:
    st.error("❌ RandomForest model missing — please add randomforest.pkl in models/")
    st.stop()
else:
    st.sidebar.success("✅ RandomForest loaded")

if cnn is not None:
    st.sidebar.success("✅ CNN model loaded")
else:
    st.sidebar.info("CNN not found")

if lstm is not None:
    st.sidebar.success("✅ LSTM model loaded")
else:
    st.sidebar.info("LSTM not found")

# Open-Meteo helpers

def fetch_open_meteo_last7(lat, lon, timezone_str="Asia/Kolkata"):
    today_utc = datetime.now(timezone.utc)
    start_dt = (today_utc - timedelta(days=7)).date()
    end_dt = today_utc.date()
    start_str = start_dt.isoformat(); end_str = end_dt.isoformat()

    api = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation",
        "start_date": start_str, "end_date": end_str, "timezone": timezone_str
    }
    try:
        r = requests.get(api, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        hourly = js.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        if len(times) == 0:
            return None, "No hourly data returned by API."
        dfh = pd.DataFrame({
            "time": times,
            "temperature_2m": hourly.get("temperature_2m", []),
            "relativehumidity_2m": hourly.get("relativehumidity_2m", []),
            "precipitation": hourly.get("precipitation", [])
        })
        dfh["date"] = dfh["time"].dt.date
        agg = dfh.groupby("date").agg({
            "temperature_2m": "mean",
            "relativehumidity_2m": "mean",
            "precipitation": "sum"
        }).reset_index().sort_values("date")
        agg = agg.tail(7)
        if agg.shape[0] < 7:
            if agg.shape[0] == 0:
                return None, "Not enough data to build 7 days."
            last = agg.iloc[-1:].copy()
            while agg.shape[0] < 7:
                agg = pd.concat([last, agg], ignore_index=True)
            agg = agg.iloc[-7:].reset_index(drop=True)
        agg = agg.rename(columns={
            "temperature_2m": "temp",
            "relativehumidity_2m": "humidity",
            "precipitation": "rain"
        })
        meta = {"start_date": agg["date"].min().isoformat(), "end_date": agg["date"].max().isoformat(),
                "fetch_time_utc": datetime.now(timezone.utc).isoformat(), "mode": "past"}
        return agg.reset_index(drop=True), meta
    except Exception as e:
        return None, str(e)

def fetch_open_meteo_next7(lat, lon, timezone_str="Asia/Kolkata"):
    """
    Fetch forecast for next 7 days (tomorrow -> +6 days) from Open-Meteo,
    aggregate hourly to daily mean temp, mean humidity, sum precipitation.
    Returns (agg_df, meta) or (None, error_string)
    """
    today_local = datetime.now().date()
    start_dt = today_local + timedelta(days=1)
    end_dt = start_dt + timedelta(days=6)
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    api = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation",
        "start_date": start_str,
        "end_date": end_str,
        "timezone": timezone_str
    }
    try:
        r = requests.get(api, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        hourly = js.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        if len(times) == 0:
            return None, "No hourly forecast returned by API."
        dfh = pd.DataFrame({
            "time": times,
            "temperature_2m": hourly.get("temperature_2m", []),
            "relativehumidity_2m": hourly.get("relativehumidity_2m", []),
            "precipitation": hourly.get("precipitation", [])
        })
        dfh["date"] = dfh["time"].dt.date
        agg = dfh.groupby("date").agg({
            "temperature_2m": "mean",
            "relativehumidity_2m": "mean",
            "precipitation": "sum"
        }).reset_index().sort_values("date")
        
        if agg.shape[0] < 7:
            if agg.shape[0] == 0:
                return None, "Not enough forecast data."
            last = agg.iloc[-1:].copy()
            while agg.shape[0] < 7:
                agg = pd.concat([agg, last], ignore_index=True)
            agg = agg.iloc[:7].reset_index(drop=True)
        else:
            agg = agg.head(7).reset_index(drop=True)
        agg = agg.rename(columns={
            "temperature_2m": "temp",
            "relativehumidity_2m": "humidity",
            "precipitation": "rain"
        })
        meta = {
            "start_date": agg["date"].min().isoformat(),
            "end_date": agg["date"].max().isoformat(),
            "fetch_time_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "forecast"
        }
        return agg.reset_index(drop=True), meta
    except Exception as e:
        return None, str(e)


st.subheader("📅 7-Day Weather Snapshot")

demo_dates = [(datetime.now() - timedelta(days=6-i)).date() for i in range(7)]
demo_data = pd.DataFrame({
    "date": demo_dates,
    "temp": [24,25,26,25,24,23,24],
    "humidity": [60,62,65,64,63,61,60],
    "rain": [0,0,1,0,0,0,0],
    "ndvi": [0.7,0.68,0.65,0.66,0.7,0.72,0.7],
    "elevation": [default_elev]*7,
})
demo_data["dryness_index"] = demo_data["temp"] - demo_data["humidity"]/5.0
X_df = demo_data[["temp","humidity","rain","ndvi","elevation","dryness_index"]].astype(float)
last_fetch_meta = None
live_connected = False
wf_display = None

if use_live:
    st.info(f"Fetching {mode} from Open-Meteo ... (may take a few seconds)")
    lat, lon = city_coords
    if mode.startswith("Past"):
        wf, meta_or_err = fetch_open_meteo_last7(lat, lon)
    else:
        wf, meta_or_err = fetch_open_meteo_next7(lat, lon)

    if wf is None:
        st.error(f"Live fetch failed: {meta_or_err}. Using demo data instead.")
        st.dataframe(demo_data, hide_index=True)
        live_connected = False
    else:
        ndvi_default = 0.6
        meta_path = ROOT / "data" / "processed" / "data_train_T7_bycity_meta.csv"
        if meta_path.exists():
            try:
                md = pd.read_csv(meta_path)
                if "nearest_city" in md.columns:
                    city_mean_ndvi = md[md["nearest_city"]==city]["ndvi"].mean()
                    if pd.notna(city_mean_ndvi):
                        ndvi_default = float(city_mean_ndvi)
            except Exception:
                pass
        wf["ndvi"] = [ndvi_default]*len(wf)
        wf["elevation"] = [cities[city]["elev"]]*len(wf)
        wf["dryness_index"] = wf["temp"] - wf["humidity"]/5.0
        wf_display = wf[["date","temp","humidity","rain","ndvi","elevation","dryness_index"]].copy()
        st.success(f"Live fetch successful: {meta_or_err['start_date']} → {meta_or_err['end_date']} (fetched {meta_or_err['fetch_time_utc']})")
        st.dataframe(wf_display, hide_index=True)
        X_df = wf[["temp","humidity","rain","ndvi","elevation","dryness_index"]].astype(float).reset_index(drop=True)
        last_fetch_meta = meta_or_err
        live_connected = True
else:
    st.dataframe(demo_data, hide_index=True)
    live_connected = False

# sidebar connection status
if use_live:
    if live_connected:
        st.sidebar.success(f"Live: OK ({last_fetch_meta['start_date']}→{last_fetch_meta['end_date']})")
    else:
        st.sidebar.error("Live: Failed — using demo data")


# Manual NDVI / Elevation override form

st.sidebar.markdown("---")
st.sidebar.header("📥 Manual Overrides (optional)")
with st.sidebar.expander("Override NDVI / Elevation (quick)"):
    use_override = st.checkbox("Enable manual override", value=False)
    ndvi_override_single = st.number_input("NDVI (single value, 0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.01, format="%.2f")
    elev_override_single = st.number_input("Elevation (single value, meters)", min_value=0.0, max_value=5000.0, value=float(default_elev), step=1.0)
    st.markdown("OR provide per-day NDVI (7 comma-separated values):")
    ndvi_override_list_raw = st.text_input("NDVI list (comma-separated, oldest→newest)", value="")
    apply_overrides = st.button("Apply overrides")


X_df_work = X_df.copy()

if use_override and apply_overrides:
    applied = False
    if ndvi_override_list_raw.strip():
        try:
            vals = [float(v.strip()) for v in ndvi_override_list_raw.split(",")]
            if len(vals) == X_df_work.shape[0]:
                X_df_work["ndvi"] = vals
                applied = True
            else:
                st.sidebar.error(f"NDVI list length ({len(vals)}) != {X_df_work.shape[0]} rows.")
        except Exception:
            st.sidebar.error("Could not parse NDVI list. Use numbers separated by commas.")
    if not applied:
        
        if use_override:
            X_df_work["ndvi"] = ndvi_override_single
            X_df_work["elevation"] = elev_override_single
            applied = True
    if applied:
        X_df_work["dryness_index"] = X_df_work["temp"] - (X_df_work["humidity"] / 5.0)
        st.sidebar.success("Overrides applied to working dataset.")
        st.write("### Adjusted 7-day table (after overrides)")
        if wf_display is not None and "date" in wf_display.columns:
            
            disp = wf_display[["date"]].copy()
            disp.reset_index(drop=True, inplace=True)
            disp[["temp","humidity","rain","ndvi","elevation","dryness_index"]] = X_df_work[["temp","humidity","rain","ndvi","elevation","dryness_index"]].values
            st.dataframe(disp, hide_index=True)
        else:
            
            demo_disp = demo_data.copy()
            demo_disp[["temp","humidity","rain","ndvi","elevation","dryness_index"]] = X_df_work[["temp","humidity","rain","ndvi","elevation","dryness_index"]].values
            st.dataframe(demo_disp, hide_index=True)


if X_df_work is None:
    X_df_work = X_df.copy()

#Prediction

x_seq = X_df_work.values.reshape(1, X_df_work.shape[0], X_df_work.shape[1])
x_flat = X_df_work.values.reshape(1, -1)

rf_p = float(rf.predict_proba(x_flat)[:, 1][0])
cnn_p = float(cnn.predict(x_seq, verbose=0)[0, 0]) if cnn else 0.0
lstm_p = float(lstm.predict(x_seq, verbose=0)[0, 0]) if lstm else 0.0

total_w = w_rf + w_cnn + w_lstm
if total_w == 0:
    st.error("Weights sum to zero — adjust sliders.")
    st.stop()

p_ens = (w_rf*rf_p + w_cnn*cnn_p + w_lstm*lstm_p) / total_w
pred = int(p_ens >= thr)

if p_ens > 0.7:
    label, color = "🔥 High Risk", "red"
elif p_ens > thr:
    label, color = "⚠️ Medium Risk", "orange"
else:
    label, color = "🌿 Low Risk", "green"


# Trend Charts (7-day)

st.subheader("📈 7-Day Trends")


if wf_display is not None and "date" in wf_display.columns:
    dates = pd.to_datetime(wf_display["date"])
else:
    
    try:
        base_dates = pd.date_range(end=pd.Timestamp.now().date(), periods=X_df_work.shape[0]).date
        dates = pd.to_datetime(base_dates)
    except Exception:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=X_df_work.shape[0])

plot_df = pd.DataFrame({
    "date": dates,
    "temp": X_df_work["temp"].values,
    "humidity": X_df_work["humidity"].values,
    "rain": X_df_work["rain"].values,
    "dryness_index": X_df_work["dryness_index"].values,
    "ndvi": X_df_work["ndvi"].values
})
plot_df = plot_df.set_index("date")
# weather plots
st.markdown("**Weather:** temperature, humidity, precipitation")
st.line_chart(plot_df[["temp", "humidity", "rain"]])

# dryness index
st.markdown("**Dryness index (temp - humidity/5)**")
st.line_chart(plot_df[["dryness_index"]])

# NDVI plot (separate — often useful to see vegetation changes)
st.markdown("**NDVI (vegetation index)**")
st.line_chart(plot_df[["ndvi"]])

# Summary info
st.info(f"Ensemble probability: **{p_ens:.3f}** → **{label}** (threshold {thr:.2f})")

def log_run(csv_path, row):
    header = ["timestamp_utc","city","mode","live_ok","start_date","end_date",
              "w_rf","w_cnn","w_lstm","threshold","probability","prediction","label",
              "ndvi_override","elevation_override"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

# Log run
log_path = RESULTS_DIR / "live_runs.csv"
ndvi_override_val = ndvi_override_single if (use_override and apply_overrides) else ""
elev_override_val = elev_override_single if (use_override and apply_overrides) else ""
meta_start = last_fetch_meta["start_date"] if last_fetch_meta else ""
meta_end = last_fetch_meta["end_date"] if last_fetch_meta else ""
log_row = [
    datetime.now(timezone.utc).isoformat(),
    city,
    (last_fetch_meta["mode"] if last_fetch_meta else "demo"),
    bool(live_connected),
    meta_start, meta_end,
    float(w_rf), float(w_cnn), float(w_lstm), float(thr),
    float(p_ens), int(pred), label,
    ndvi_override_val, elev_override_val
]
try:
    log_run(str(log_path), log_row)
except Exception:
    pass

#Display Results

st.subheader("📊 Ensemble Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Probability", f"{p_ens:.3f}")
col2.metric("Threshold", f"{thr:.2f}")
col3.metric("Risk Level", label)

if st.checkbox("Show individual model outputs"):
    st.table(pd.DataFrame({
        "Model": ["RandomForest", "CNN", "LSTM"],
        "Probability": [rf_p, cnn_p, lstm_p]
    }))

#Map Display

st.subheader("🗺️ Risk Location Map")
lat, lon = city_coords
m = folium.Map(location=[lat, lon], zoom_start=8)
folium.CircleMarker(
    [lat, lon],
    radius=12,
    color=color,
    fill=True,
    fill_color=color,
    fill_opacity=0.8,
    popup=f"{city}: {label} ({p_ens:.2f})"
).add_to(m)
st_folium(m, width=700, height=450)

# Model Performance Summary

st.markdown("---")
st.subheader("📈 Model Performance Summary")

conf_img = PLOTS_DIR / "ensemble_confusion_thr0.40.png"
if conf_img.exists():
    st.image(str(conf_img), caption="Final Ensemble Confusion Matrix @ thr=0.40")
else:
    st.info("No confusion matrix found yet.")

pred_csv = RESULTS_DIR / "ensemble_final_predictions_thr0.40.csv"
if pred_csv.exists():
    with open(pred_csv, "rb") as f:
        st.download_button("⬇️ Download Final Predictions CSV", f, file_name=pred_csv.name)
else:
    st.warning("No predictions CSV found in results/")

st.markdown("---")
if live_connected:
    st.caption(f"Optimized Ensemble: RF(1.0), CNN(0.3), LSTM(0.3) • Threshold = 0.40 • Live data used for dates {last_fetch_meta['start_date']} → {last_fetch_meta['end_date']}")
else:
    st.caption("Optimized Ensemble: RF(1.0), CNN(0.3), LSTM(0.3) • Threshold = 0.40 • Using demo data")
