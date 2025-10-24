# src/data/resample_dem.py
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

SRC = Path('data/raw/terrain/srtm_uttarakhand_90m.tif')
DST = Path('data/raw/terrain/srtm_uttarakhand_1000m.tif')

if not SRC.exists():
    raise SystemExit(f"Source DEM not found: {SRC}")

print("Resampling DEM to 1000m (EPSG:3857). This may take a minute...")
with rasterio.open(SRC) as src:
    dst_crs = 'EPSG:3857'
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=1000)
    kwargs = src.meta.copy()
    kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
    with rasterio.open(DST, 'w', **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.average)
print("Wrote resampled DEM:", DST)
