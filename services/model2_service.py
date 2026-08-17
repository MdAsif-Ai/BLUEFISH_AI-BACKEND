"""
BlueFish AI - Model 2 Service: Ocean Fronts & Eddies Detection
===============================================================
Wraps the pure-math ocean front and eddy detection module.

Model 2 uses:
  - Sobel gradient filtering on SST fields → detects thermal fronts
  - Okubo-Weiss parameter (W) on velocity fields → detects eddies

Input: xarray Dataset with variables: sst, uo (u-velocity), vo (v-velocity), zos/ssh
Output: GeoJSON FeatureCollection with detected fronts and eddies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label, sobel, zoom
try:
    from skimage.transform import resize
except ImportError:
    def resize(image, output_shape, **kwargs):
        zoom_factors = [o / s for o, s in zip(output_shape, image.shape)]
        return zoom(image, zoom_factors, order=1)

logger = logging.getLogger("bluefish.services.model2")


class OceanFrontEddyModel:
    """
    BlueFish AI - Model 2: Ocean Front & Eddy Detector
    Production-grade estimator with optimized oceanographic thresholds.
    """

    def __init__(
        self,
        gaussian_sigma: float = 1.0,
        front_percentile: float = 90.0,
        min_eddy_pixels: int = 4,
        km_per_degree: float = 111.0,
    ):
        self.gaussian_sigma = gaussian_sigma
        self.front_percentile = front_percentile
        self.min_eddy_pixels = min_eddy_pixels
        self.km_per_degree = km_per_degree

    def _get_var(self, ds: Any, names: List[str]) -> Any:
        for n in names:
            if n in ds.variables:
                return ds[n]
        raise ValueError(f"None of the variables {names} were found in the dataset.")

    def _preprocess(self, ds: Any) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sst_da = self._get_var(ds, ["sst", "thetao", "analysed_sst"])
        uo_da = self._get_var(ds, ["uo"])
        vo_da = self._get_var(ds, ["vo"])

        ssh_da = None
        try:
            ssh_da = self._get_var(ds, ["zos", "ssh"])
        except ValueError:
            ssh_da = None

        if "depth" in sst_da.dims:
            sst_da = sst_da.isel(depth=0)
        if "depth" in uo_da.dims:
            uo_da = uo_da.isel(depth=0)
        if "depth" in vo_da.dims:
            vo_da = vo_da.isel(depth=0)
        if ssh_da is not None and "depth" in ssh_da.dims:
            ssh_da = ssh_da.isel(depth=0)

        sst_grid = sst_da.values
        uo_grid = uo_da.values
        vo_grid = vo_da.values
        ssh_grid = ssh_da.values if ssh_da is not None else None

        target_shape = sst_grid.shape
        if uo_grid.shape != target_shape:
            uo_grid = resize(uo_grid, target_shape, anti_aliasing=True, preserve_range=True)
        if vo_grid.shape != target_shape:
            vo_grid = resize(vo_grid, target_shape, anti_aliasing=True, preserve_range=True)
        if ssh_grid is not None and ssh_grid.shape != target_shape:
            ssh_grid = resize(ssh_grid, target_shape, anti_aliasing=True, preserve_range=True)

        lat_name = "latitude" if "latitude" in sst_da.coords else ("lat" if "lat" in sst_da.coords else sst_da.dims[0])
        lon_name = "longitude" if "longitude" in sst_da.coords else ("lon" if "lon" in sst_da.coords else sst_da.dims[1])
        lat_vals = sst_da[lat_name].values
        lon_vals = sst_da[lon_name].values

        return sst_grid, ssh_grid, uo_grid, vo_grid, lat_vals, lon_vals

    def _detect_fronts(self, sst_grid: np.ndarray) -> np.ndarray:
        """Optimized Front Detection using adaptive percentile threshold."""
        sst_grid = np.asarray(sst_grid, dtype=np.float32)
        valid = ~np.isnan(sst_grid)

        fill_val = np.nanmedian(sst_grid)
        if np.isnan(fill_val):
            fill_val = 0.0
        sst_filled = np.nan_to_num(sst_grid, nan=fill_val)

        sst_smooth = gaussian_filter(sst_filled, sigma=self.gaussian_sigma)

        gx = sobel(sst_smooth, axis=1)
        gy = sobel(sst_smooth, axis=0)
        grad = np.sqrt(gx**2 + gy**2)

        valid_grads = grad[valid]
        if len(valid_grads) > 0:
            threshold = np.percentile(valid_grads, self.front_percentile)
        else:
            threshold = 0

        front_mask = (grad > threshold) & valid
        return front_mask

    def _detect_eddies(
        self,
        uo_orig: np.ndarray,
        vo_orig: np.ndarray,
        ssh_orig: Optional[np.ndarray],
        lat_vals: np.ndarray,
        lon_vals: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Optimized Eddy Detection with Okubo-Weiss parameter W and SSH cross-verification."""
        valid = ~np.isnan(uo_orig) & ~np.isnan(vo_orig)
        uo = np.nan_to_num(np.asarray(uo_orig, dtype=np.float32), nan=0.0)
        vo = np.nan_to_num(np.asarray(vo_orig, dtype=np.float32), nan=0.0)

        du_dx = np.gradient(uo, axis=1)
        du_dy = np.gradient(uo, axis=0)
        dv_dx = np.gradient(vo, axis=1)
        dv_dy = np.gradient(vo, axis=0)

        s_n = du_dx - dv_dy
        s_s = dv_dx + du_dy
        vort = dv_dx - du_dy
        W = s_n**2 + s_s**2 - vort**2

        w_var = np.nanvar(W)
        if w_var > 0:
            w_threshold = -0.2 * w_var
        else:
            return []

        mask = (W < w_threshold) & valid
        labeled, num = label(mask)

        lat_step = abs(lat_vals[1] - lat_vals[0]) * self.km_per_degree if len(lat_vals) > 1 else 1.0
        lon_step = abs(lon_vals[1] - lon_vals[0]) * self.km_per_degree if len(lon_vals) > 1 else 1.0
        pixel_area = lat_step * lon_step

        eddies = []
        for i in range(1, num + 1):
            blob = labeled == i
            pixels = int(blob.sum())
            if pixels < self.min_eddy_pixels:
                continue

            rows, cols = np.where(blob)
            r, c = int(rows.mean()), int(cols.mean())
            radius = float(np.sqrt((pixels * pixel_area) / np.pi))

            vort_at_center = vort[r, c]
            is_cyclonic = vort_at_center > 0
            eddy_type = "cyclonic_cold_core" if is_cyclonic else "anticyclonic_warm_core"

            ssh_val = None
            if ssh_orig is not None and r < ssh_orig.shape[0] and c < ssh_orig.shape[1]:
                val = ssh_orig[r, c]
                if np.isfinite(val):
                    ssh_val = float(val)
                    if is_cyclonic and ssh_val > 0:
                        continue
                    if not is_cyclonic and ssh_val < 0:
                        continue

            strength = float(np.nanmean(np.abs(W[blob])))

            eddies.append({
                "center_lat": float(lat_vals[r]),
                "center_lon": float(lon_vals[c]),
                "radius_km": round(radius, 2),
                "type": eddy_type,
                "rotation": "cyclonic" if is_cyclonic else "anticyclonic",
                "strength": round(strength, 4),
                "okubo_weiss": round(strength, 4),
                "ssh_at_center": ssh_val,
                "pixel_count": pixels,
            })
        return eddies

    def _format_output(
        self,
        front_mask: np.ndarray,
        eddies: List[Dict[str, Any]],
        lat_vals: np.ndarray,
        lon_vals: np.ndarray,
    ) -> Dict[str, Any]:
        lons_grid, lats_grid = np.meshgrid(lon_vals, lat_vals)
        front_lats = lats_grid[front_mask].tolist()
        front_lons = lons_grid[front_mask].tolist()
        fronts = [
            {"lat": round(lat, 4), "lon": round(lon, 4), "gradient_magnitude": 0.8}
            for lat, lon in zip(front_lats, front_lons)
        ]
        return {"fronts": fronts, "eddies": eddies}

    def predict(self, daily_ds: Any) -> Dict[str, Any]:
        """
        Main API Method. Pass an xarray Dataset for a single day, get JSON output.
        """
        sst, ssh, uo, vo, lats, lons = self._preprocess(daily_ds)
        front_mask = self._detect_fronts(sst)
        eddies = self._detect_eddies(uo, vo, ssh, lats, lons)
        return self._format_output(front_mask, eddies, lats, lons)


class OceanFeatureService:
    """
    Wraps OceanFrontEddyModel for production use.
    Normalizes inputs to xarray Dataset format expected by model2.
    """

    def __init__(self, model2_instance: Optional[Any] = None):
        if model2_instance is None:
            self.model = OceanFrontEddyModel()
        else:
            self.model = model2_instance

    def detect_from_dataset(self, ds: Any, date_str: Optional[str] = None) -> Dict[str, Any]:
        if ds is None:
            return {"fronts": [], "eddies": [], "error": "No dataset provided"}

        try:
            result = self.model.predict(ds)
            return {
                "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "fronts": result.get("fronts", []),
                "eddies": result.get("eddies", []),
                "front_count": len(result.get("fronts", [])),
                "eddy_count": len(result.get("eddies", [])),
            }
        except Exception as e:
            logger.error(f"Ocean feature detection failed: {e}", exc_info=True)
            return {"fronts": [], "eddies": [], "error": str(e)}

    def detect(self, ds: Any, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Alias for detect_from_dataset."""
        return self.detect_from_dataset(ds, date_str)

    def detect_from_grids(
        self,
        sst_grid: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        ssh_grid: Optional[np.ndarray] = None,
        date_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience method when data is available as numpy arrays."""
        try:
            # pyrefly: ignore [missing-import]
            import xarray as xr

            data_vars = {
                "sst": (["latitude", "longitude"], sst_grid),
                "uo": (["latitude", "longitude"], u_grid),
                "vo": (["latitude", "longitude"], v_grid),
            }
            if ssh_grid is not None:
                data_vars["zos"] = (["latitude", "longitude"], ssh_grid)

            ds = xr.Dataset(
                data_vars,
                coords={"latitude": lat_coords, "longitude": lon_coords},
            )
            return self.detect_from_dataset(ds, date_str)
        except Exception as e:
            logger.error(f"Grid-to-dataset conversion failed: {e}")
            return {"fronts": [], "eddies": [], "error": str(e)}

    def to_geojson(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        features = []

        for front in detection_result.get("fronts", []):
            lat = front.get("lat")
            lon = front.get("lon")
            if lat is not None and lon is not None:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {
                        "type": "front",
                        "gradient_magnitude": front.get("gradient_magnitude", 0.0),
                    },
                })

        for eddy in detection_result.get("eddies", []):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [eddy.get("center_lon", eddy.get("lon", 0.0)), eddy.get("center_lat", eddy.get("lat", 0.0))],
                },
                "properties": {
                    "type": "eddy",
                    "rotation": eddy.get("rotation", "cyclonic"),
                    "radius_km": eddy.get("radius_km", 50.0),
                    "okubo_weiss": eddy.get("okubo_weiss", 0.0),
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "date": detection_result.get("date"),
            "meta": {
                "front_count": detection_result.get("front_count", 0),
                "eddy_count": detection_result.get("eddy_count", 0),
            },
        }
