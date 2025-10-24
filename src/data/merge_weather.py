# src/data/merge_weather.py
import pandas as pd
from pathlib import Path

RAW = Path('data/raw/weather')
OUT = Path('data/processed')
OUT.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RAW.glob('*.csv'))
if not csv_files:
    raise SystemExit(f"No CSV weather files found in {RAW}. Place Dehradun.csv etc. there.")

dfs = []
for f in csv_files:
    city_name = f.stem.strip().lower().replace(' ', '_')
    df = pd.read_csv(f)
    # drop stray unnamed index
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.columns = [c.strip().lower() for c in df.columns]
    # find datetime column
    dt_col = None
    for cand in ['date','datetime','time','timestamp']:
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        # try heuristics
        for c in df.columns:
            if 'date' in c or 'time' in c:
                dt_col = c; break
    if dt_col is None:
        raise SystemExit(f"Could not find datetime column in {f.name}. Columns: {df.columns.tolist()}")

    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
    df = df[~df[dt_col].isna()].copy()
    if 'city' not in df.columns:
        df['city'] = city_name
    else:
        df['city'] = df['city'].astype(str).str.strip().str.lower()

    # normalize likely columns
    colmap = {}
    for c in df.columns:
        if 'temp' in c and 'min' not in c:
            colmap[c] = 'temperature_2m'
        if 'humid' in c and 'relative' in c:
            colmap[c] = 'relative_humidity_2m'
        if c == 'dew_point_2m':
            colmap[c] = 'dew_point_2m'
        if 'precip' in c or c == 'rain':
            colmap[c] = 'precipitation'
        if 'wind' in c and 'speed' in c:
            colmap[c] = 'wind_speed'
    df = df.rename(columns=colmap)

    # ensure canonical cols exist
    for need in ['temperature_2m','relative_humidity_2m','precipitation','wind_speed']:
        if need not in df.columns:
            df[need] = pd.NA

    # aggregate to daily per city (mean)
    df['date'] = df[dt_col].dt.tz_convert(None).dt.date if df[dt_col].dt.tz is not None else df[dt_col].dt.date
    daily = df.groupby(['city','date'], as_index=False)[['temperature_2m','relative_humidity_2m','precipitation','wind_speed']].mean()
    dfs.append(daily)

weather_all = pd.concat(dfs, ignore_index=True, sort=False)
weather_all = weather_all.sort_values(['city','date']).reset_index(drop=True)
out_path = OUT / 'weather_uttarakhand_combined.csv'
weather_all.to_csv(out_path, index=False)
print("Saved combined daily weather to:", out_path)
print(weather_all.head(6).to_string(index=False))
