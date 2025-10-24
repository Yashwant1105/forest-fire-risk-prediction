import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path('data/processed')
IN = PROC/'forest_fire_dataset_uttarakhand.csv'
OUT_NPZ = PROC/'data_train_T7_bycity.npz'
OUT_META = PROC/'data_train_T7_bycity_meta.csv'

if not IN.exists():
    raise SystemExit(f"Input not found: {IN}. Run prepare_dataset.py first.")

print("Loading dataset:", IN)
df = pd.read_csv(IN)
print("Rows:", len(df))

df['date'] = pd.to_datetime(df['date'])
required_city_col = 'nearest_city'
if required_city_col not in df.columns:
    raise SystemExit(f"Column {required_city_col} not found in dataset.")

features = ['temp', 'humidity', 'rain', 'ndvi', 'elevation', 'dryness_index']

# 1) Fill NDVI missing values with city mean 
if df['ndvi'].isna().sum() > 0:
    print("NDVI missing:", df['ndvi'].isna().sum(), "-> filling with city mean")
    df['ndvi'] = df.groupby('nearest_city')['ndvi'].transform(lambda x: x.fillna(x.mean()))
    df['ndvi'] = df['ndvi'].fillna(df['ndvi'].mean())

# 2) Impute weather: for each city, sort by date and ffill/bfill, then fill remaining with city mean
for col in ['temp', 'humidity', 'rain']:
    miss_before = df[col].isna().sum()
    if miss_before == 0:
        print(f"{col}: no missing")
        continue
    print(f"{col}: missing before = {miss_before}; imputing by city forward/backfill then mean")
    # Keep original order index
    df = df.sort_values(['nearest_city','date']).reset_index(drop=False)
    df[col] = df.groupby('nearest_city')[col].transform(lambda g: g.ffill().bfill())
    df[col] = df.groupby('nearest_city')[col].transform(lambda g: g.fillna(g.mean()))
    df[col] = df[col].fillna(df[col].mean())
    df = df.set_index('index').sort_index().reset_index(drop=True)
    print(f"{col}: missing after = {df[col].isna().sum()}")

# 3) Recompute dryness_index
df['dryness_index'] = df['temp'] - (df['humidity'] / 5.0)

# 4) Create a date-only column and aggregate by nearest_city + date_only (daily mean)
df['date_only'] = df['date'].dt.date
agg = df.groupby(['nearest_city', 'date_only'], as_index=False).agg({
    'temp':'mean','humidity':'mean','rain':'mean','ndvi':'mean','elevation':'mean','dryness_index':'mean','fire':'mean'
})
agg['date'] = pd.to_datetime(agg['date_only'])

# 5) For each city build sequences T=7
T = 7
seqs, labels, meta = [], [], []
cities_kept = []
for city, g in agg.groupby('nearest_city'):
    g = g.sort_values('date').reset_index(drop=True)
    # drop rows that still have NaNs in features (should be none)
    g = g.dropna(subset=features)
    if len(g) < T:
        continue
    vals = g[features].values
    labs = (g['fire'].values >= 0.5).astype(int)
    dates = g['date'].values
    # approximate lat/lon per city = mean from original df
    city_coords = df[df['nearest_city'] == city][['latitude','longitude']].dropna()
    mean_lat = city_coords['latitude'].mean() if not city_coords.empty else np.nan
    mean_lon = city_coords['longitude'].mean() if not city_coords.empty else np.nan
    for i in range(len(g)-T+1):
        seqs.append(vals[i:i+T])
        labels.append(int(labs[i+T-1]))
        meta.append((city, str(np.datetime_as_string(dates[i+T-1], unit='D')), float(mean_lat), float(mean_lon)))
    cities_kept.append((city, len(g)))

X = np.array(seqs)
y = np.array(labels)

print("Cities with sequences (city, days):", cities_kept)
print("Final sequences shape:", X.shape, "Labels shape:", y.shape)
print("Positive fraction:", (y.sum()/len(y) if len(y)>0 else 0.0))

# Save
np.savez_compressed(OUT_NPZ, X=X, y=y)
pd.DataFrame(meta, columns=['city','date','lat','lon']).to_csv(OUT_META, index=False)
print("Saved sequences to:", OUT_NPZ)
print("Saved meta to:", OUT_META)
