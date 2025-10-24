import glob, numpy as np, pandas as pd
from pathlib import Path
import rasterio
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings('ignore')

ROOT = Path.cwd()
RAW = ROOT/'data'/'raw'
PROC = ROOT/'data'/'processed'
PROC.mkdir(parents=True, exist_ok=True)

# 1) Read FIRMS files
fire_files = sorted(glob.glob(str(RAW/'fires'/'*.csv')))
if not fire_files:
    raise SystemExit("No FIRMS CSV files found in data/raw/fires/. Place modis_*.csv there.")
dfs = [pd.read_csv(f) for f in fire_files]
fires = pd.concat(dfs, ignore_index=True, sort=False)
fires.columns = [c.strip().lower() for c in fires.columns]

# check expected cols
for need in ['latitude','longitude','acq_date']:
    if need not in fires.columns:
        raise SystemExit(f"FIRMS files missing required column: {need}. Found columns: {fires.columns.tolist()}")

# parse date
fires['acq_date'] = pd.to_datetime(fires['acq_date'], errors='coerce')
fires = fires.dropna(subset=['acq_date']).copy()
fires['date'] = fires['acq_date'].dt.date

# bbox for Uttarakhand (slightly expanded)
lat_min, lat_max = 28.0, 32.0
lon_min, lon_max = 76.5, 81.5
fires = fires[(fires['latitude']>=lat_min)&(fires['latitude']<=lat_max)&(fires['longitude']>=lon_min)&(fires['longitude']<=lon_max)].copy()
fires.reset_index(drop=True, inplace=True)
print("Fire points after bbox filter:", len(fires))

# 2) Load weather aggregated daily
weather_path = PROC/'weather_uttarakhand_combined.csv'
if not weather_path.exists():
    raise SystemExit(f"Weather file not found: {weather_path}. Run merge_weather.py first.")
weather = pd.read_csv(weather_path)
weather.columns = [c.strip().lower() for c in weather.columns]
weather['date'] = pd.to_datetime(weather['date']).dt.date

# pick weather columns
temp_col = 'temperature_2m' if 'temperature_2m' in weather.columns else next((c for c in weather.columns if 'temp' in c), None)
hum_col = 'relative_humidity_2m' if 'relative_humidity_2m' in weather.columns else next((c for c in weather.columns if 'humid' in c), None)
rain_col = 'precipitation' if 'precipitation' in weather.columns else next((c for c in weather.columns if 'precip' in c or c=='rain'), None)
print("Weather columns used:", temp_col, hum_col, rain_col)

# build city coords (if weather contains lat/lon)
cities_found = sorted(weather['city'].dropna().unique().tolist())
city_coords = {}
if 'lat' in weather.columns and 'lon' in weather.columns:
    for c in cities_found:
        r = weather[weather['city']==c]
        if not r.empty:
            city_coords[c] = (float(r.iloc[0]['lat']), float(r.iloc[0]['lon']))

# fallback coords for common cities
fallback = {'dehradun':(30.3165,78.0322),'nainital':(29.3919,79.4542),'almora':(29.5942,79.6576),'haridwar':(29.9457,78.1642)}
for c in cities_found:
    if c not in city_coords:
        key=c.lower()
        if key in fallback:
            city_coords[c]=fallback[key]

if not city_coords:
    # if weather city names differ, build city_coords from fallback keys present
    for k in fallback:
        if k in cities_found:
            city_coords[k]=fallback[k]

city_list = list(city_coords.keys())
coords_arr = np.array([city_coords[c] for c in city_list])
if len(coords_arr)==0:
    raise SystemExit("No city coordinates available for nearest-city matching. Check weather CSV 'city' values.")
kdt = KDTree(coords_arr, metric='euclidean')

# 3) Attach nearest-city weather to fires
fire_coords = np.vstack([fires['latitude'].values, fires['longitude'].values]).T
dists, idxs = kdt.query(fire_coords, k=1)
fires['nearest_city'] = [city_list[i] for i in idxs.ravel()]
weather_for_merge = weather.rename(columns={'city':'nearest_city'})
fires = fires.merge(weather_for_merge, how='left', on=['date','nearest_city'])
print("Missing temperature after merge:", fires[temp_col].isna().sum() if temp_col in fires.columns else 'temp missing')

# 4) Sample NDVI and DEM rasters
ndvi_path = RAW/'ndvi'/'ndvi_uttarakhand_2015_2024.tif'
dem_90 = RAW/'terrain'/'srtm_uttarakhand_90m.tif'
dem_1000 = RAW/'terrain'/'srtm_uttarakhand_1000m.tif'
if not ndvi_path.exists():
    raise SystemExit(f"NDVI raster missing: {ndvi_path}")

def sample_raster(raster_path, lons, lats):
    vals = np.full(len(lons), np.nan, dtype=float)
    with rasterio.open(str(raster_path)) as src:
        coords = list(zip(lons, lats))
        for i, v in enumerate(src.sample(coords)):
            vals[i] = float(v[0]) if v is not None else np.nan
    return vals

print("Sampling NDVI at fire points...")
fires['ndvi_raw'] = sample_raster(ndvi_path, fires['longitude'].values, fires['latitude'].values)
if np.nanmax(fires['ndvi_raw']) > 2:
    fires['ndvi'] = fires['ndvi_raw'] / 10000.0
else:
    fires['ndvi'] = fires['ndvi_raw']

dem_use = dem_1000 if dem_1000.exists() else (dem_90 if dem_90.exists() else None)
if dem_use:
    print("Sampling DEM:", dem_use)
    fires['elevation'] = sample_raster(dem_use, fires['longitude'].values, fires['latitude'].values)
else:
    fires['elevation'] = np.nan

# 5) features & label
fires['temp'] = fires.get(temp_col, np.nan)
fires['humidity'] = fires.get(hum_col, np.nan)
fires['rain'] = fires.get(rain_col, np.nan)
fires['dryness_index'] = fires['temp'] - (fires['humidity'] / 5.0)
fires['fire'] = 1

out_points = PROC/'forest_fire_points_uttarakhand.csv'
fires.to_csv(out_points, index=False)
print("Saved point-level merged CSV:", out_points)

# 6) create negatives (balanced)
all_dates = pd.date_range(fires['date'].min(), fires['date'].max()).date
n_pos = len(fires)
rng = np.random.default_rng(42)
neg_samples=[]
attempts=0
while len(neg_samples) < n_pos and attempts < n_pos*12:
    attempts+=1
    lat = rng.uniform(lat_min, lat_max)
    lon = rng.uniform(lon_min, lon_max)
    d = rng.choice(all_dates)
    near = fires[(abs(fires['latitude']-lat)<0.02) & (abs(fires['longitude']-lon)<0.02) & (fires['date']==d)]
    if len(near)==0:
        neg_samples.append((lat, lon, d))
neg_df = pd.DataFrame(neg_samples, columns=['latitude','longitude','date'])
neg_df['date'] = pd.to_datetime(neg_df['date']).dt.date
neg_df['nearest_city'] = [city_list[kdt.query([[lat,lon]], k=1)[1][0][0]] for lat,lon,_ in neg_samples]
neg_df = neg_df.merge(weather_for_merge, how='left', left_on=['date','nearest_city'], right_on=['date','nearest_city'])
neg_df['ndvi_raw'] = sample_raster(ndvi_path, neg_df['longitude'].values, neg_df['latitude'].values)
if np.nanmax(neg_df['ndvi_raw']) > 2:
    neg_df['ndvi'] = neg_df['ndvi_raw']/10000.0
else:
    neg_df['ndvi'] = neg_df['ndvi_raw']
if dem_use:
    neg_df['elevation'] = sample_raster(dem_use, neg_df['longitude'].values, neg_df['latitude'].values)
else:
    neg_df['elevation'] = np.nan
neg_df['temp'] = neg_df.get(temp_col, np.nan)
neg_df['humidity'] = neg_df.get(hum_col, np.nan)
neg_df['rain'] = neg_df.get(rain_col, np.nan)
neg_df['dryness_index'] = neg_df['temp'] - (neg_df['humidity']/5.0)
neg_df['fire'] = 0

# 7) combine & save
cols_keep = ['latitude','longitude','date','nearest_city','temp','humidity','rain','ndvi','elevation','dryness_index','fire']
pos_df = fires.rename(columns={'ndvi':'ndvi'})[cols_keep]
neg_df2 = neg_df[cols_keep]
full = pd.concat([pos_df, neg_df2], ignore_index=True)
out_full = PROC/'forest_fire_dataset_uttarakhand.csv'
full.to_csv(out_full, index=False)
print("Saved final tabular dataset:", out_full)
print("Counts:", full['fire'].value_counts())

# 8) create sequences T=7
full['date'] = pd.to_datetime(full['date'])
full['cell_id'] = (full['latitude'].round(3).astype(str) + '_' + full['longitude'].round(3).astype(str))
features = ['temp','humidity','rain','ndvi','elevation','dryness_index']
T = 7
seqs, labels, meta = [], [], []
full = full.sort_values(['cell_id','date']).reset_index(drop=True)
for cell, g in full.groupby('cell_id'):
    g = g.dropna(subset=features)
    if len(g) < T: continue
    vals = g[features].values
    labs = g['fire'].values
    dates = g['date'].values
    lats = g['latitude'].values
    lons = g['longitude'].values
    for i in range(len(g)-T+1):
        seqs.append(vals[i:i+T])
        labels.append(int(labs[i+T-1]))
        meta.append((cell, str(dates[i+T-1].date()), float(lats[i+T-1]), float(lons[i+T-1])))
X = np.array(seqs); y = np.array(labels)
np.savez_compressed(PROC/'data_train_T7.npz', X=X, y=y)
pd.DataFrame(meta, columns=['cell_id','date','lat','lon']).to_csv(PROC/'data_train_meta.csv', index=False)
print("Saved sequences:", PROC/'data_train_T7.npz', "shape:", X.shape, y.shape)
