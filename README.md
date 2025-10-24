# 🌲 Forest Fire Risk Prediction — Uttarakhand Hill Regions

**End-to-end Machine Learning & Deep Learning project** for predicting forest fire risk using multi-source geospatial and weather data.  
Built with **Python, TensorFlow, Scikit-learn, Streamlit, and remote sensing data (NDVI, DEM, Weather)**.

---

## 🧠 Overview

Forest fires in hilly regions like **Uttarakhand** are influenced by temperature, humidity, vegetation, rainfall, and terrain.  
This project builds an **ensemble deep learning system** that predicts forest fire risk at a fine spatiotemporal scale by combining:

- 🌡️ **Weather features** (temperature, humidity, precipitation)  
- 🌿 **Vegetation index (NDVI)** from MODIS satellite  
- 🏔️ **Elevation** from SRTM DEM  
- ⏱️ **Temporal patterns** (past 7-day sequences)

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Data Handling** | Pandas, NumPy, GeoPandas |
| **Geospatial** | Rasterio, Shapely, Folium |
| **ML / DL** | TensorFlow (CNN, LSTM), Scikit-learn (RF) |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Data Sources** | MODIS, Open-Meteo API, SRTM DEM |

---

## 📊 Model Pipeline

### 1️⃣ Data Preparation  
- Hourly **weather data** for Almora, Dehradun, and Haridwar (2010–2024).  
- **MODIS Active Fire Points** to label fire/non-fire samples.  
- **SRTM Elevation** and **NDVI layers** sampled at each coordinate.  
- Created 7-day rolling **temporal sequences** per location.

### 2️⃣ Model Training  
| Model | Description | AUC |
|--------|--------------|------|
| **Random Forest** | Baseline tabular model | 0.86 |
| **1D CNN** | Temporal weather–NDVI patterns | 0.74 |
| **LSTM** | Sequence modeling (7-day window) | 0.73 |
| **Ensemble (RF + CNN + LSTM)** | Weighted combination | **0.864** |

### 3️⃣ Final Ensemble  
Best weights (by F1 score):  
> **RF: 1.0, CNN: 0.3, LSTM: 0.3**

---

## 🧩 Results

| Metric | Value |
|--------|--------|
| **AUC** | 0.864 |
| **Accuracy** | 0.79 |
| **Precision** | 0.62 |
| **Recall** | 0.77 |
| **F1-score** | 0.69 |

**Confusion Matrix (threshold = 0.4):**
[[1204, 306],
[148, 506]]


✅ Ensemble achieved the highest overall performance.

---

## 🌍 Data Sources

| Source | Description |
|--------|-------------|
| [MODIS Active Fire (MCD14DL)](https://firms.modaps.eosdis.nasa.gov/download/) | Fire occurrence points |
| [Open-Meteo API](https://open-meteo.com/) | Hourly weather data |
| [MODIS NDVI (MOD13A2)](https://lpdaac.usgs.gov/products/mod13a2v061/) | Vegetation index |
| [SRTM DEM (NASA)](https://www2.jpl.nasa.gov/srtm/) | Elevation data |

---

## 🧮 Repository Structure
```
forest-fire-risk-prediction/
├── data/
│ └── sample/ ← demo data
├── src/
│ ├── app/ ← Streamlit frontend
│ │ └── forest_fire_app.py
│ ├── data/ ← preprocessing scripts
│ └── models/ ← ML & ensemble scripts
│ └── ensemble_final_sweep.py
├── plots/ ← result visualizations
├── requirements.txt
└── .gitignore
```

---

## 🚀 Run Locally

### 1️⃣ Clone and Set Up Environment
```bash
git clone https://github.com/Yashwant1105/forest-fire-risk-prediction.git
cd forest-fire-risk-prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App
```bash
streamlit run src/app/forest_fire_app.py
```
🖥️ Open the provided URL (usually ```http://localhost:8501```)
to access the dashboard.

The app fetches live weather data (via Open-Meteo)
and predicts forest fire risk probabilities.

### 🧠 Model Training (Optional)
To re-train models and ensemble:
```bash
python src/data/prepare_dataset.py
python src/data/create_sequences_by_city.py
python src/models/ensemble_final_sweep.py
```

## 🏆 Key Highlights

🧩 End-to-end ML pipeline: from raw data → ensemble model → web app

🌍 Combines deep learning and classical ML

📈 Achieves ``AUC ≈ 0.864`` on real regional data

🧭 Designed for reproducibility & research publication

## 📈 Future Work

🔁 Integrate real-time NDVI via Google Earth Engine API

🌬️ Add wind speed and drought indices

☁️ Deploy app via AWS Lambda or HuggingFace Spaces

🗺️ Add spatial clustering for early alert systems

