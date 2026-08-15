# model2_service.py
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, sobel, label
from skimage.transform import resize
from typing import Dict, List, Any

class OceanFrontEddyModel:
    """
    BlueFish AI - Model 2: Ocean Front & Eddy Detector
    Production-grade estimator with optimized oceanographic thresholds.
    """
    
    def __init__(self, gaussian_sigma: float = 1.0, front_percentile: float = 90.0, 
                 min_eddy_pixels: int = 4, km_per_degree: float = 111.0):
        """
        Args:
            gaussian_sigma: Smoothing factor before gradient detection.
            front_percentile: Top percentile of gradients considered as fronts (90 = Top 10%).
            min_eddy_pixels: Minimum size of an eddy to filter out noise.
        """
        self.gaussian_sigma = gaussian_sigma
        self.front_percentile = front_percentile
        self.min_eddy_pixels = min_eddy_pixels
        self.km_per_degree = km_per_degree

    def _get_var(self, ds: xr.Dataset, names: List[str]) -> xr.DataArray:
        for n in names:
            if n in ds.variables:
                return ds[n]
        raise ValueError(f"None of the variables {names} were found in the dataset.")

    def _preprocess(self, ds: xr.Dataset) -> tuple:
        sst_da = self._get_var(ds, ['sst', 'thetao', 'analysed_sst'])
        ssh_da = self._get_var(ds, ['zos', 'ssh'])
        uo_da = self._get_var(ds, ['uo'])
        vo_da = self._get_var(ds, ['vo'])

        if 'depth' in sst_da.dims: sst_da = sst_da.isel(depth=0)
        if 'depth' in uo_da.dims: uo_da = uo_da.isel(depth=0)
        if 'depth' in vo_da.dims: vo_da = vo_da.isel(depth=0)

        sst_grid = sst_da.values
        ssh_grid = ssh_da.values
        uo_grid = uo_da.values
        vo_grid = vo_da.values

        # Force grids to match using Scikit-Image Resize
        target_shape = sst_grid.shape
        if ssh_grid.shape != target_shape:
            ssh_grid = resize(ssh_grid, target_shape, anti_aliasing=True, preserve_range=True)
        if uo_grid.shape != target_shape:
            uo_grid = resize(uo_grid, target_shape, anti_aliasing=True, preserve_range=True)
        if vo_grid.shape != target_shape:
            vo_grid = resize(vo_grid, target_shape, anti_aliasing=True, preserve_range=True)

        lat_name = "latitude" if "latitude" in sst_da.coords else "lat"
        lon_name = "longitude" if "longitude" in sst_da.coords else "lon"
        lat_vals = sst_da[lat_name].values
        lon_vals = sst_da[lon_name].values

        return sst_grid, ssh_grid, uo_grid, vo_grid, lat_vals, lon_vals

    def _detect_fronts(self, sst_grid: np.ndarray) -> np.ndarray:
        """Optimized Front Detection using adaptive percentile threshold."""
        sst_grid = np.asarray(sst_grid, dtype=np.float32)
        valid = ~np.isnan(sst_grid)
        
        fill_val = np.nanmedian(sst_grid)
        if np.isnan(fill_val): fill_val = 0.0
        sst_filled = np.nan_to_num(sst_grid, nan=fill_val)
        
        sst_smooth = gaussian_filter(sst_filled, sigma=self.gaussian_sigma)
        
        gx = sobel(sst_smooth, axis=1)
        gy = sobel(sst_smooth, axis=0)
        grad = np.sqrt(gx**2 + gy**2)
        
        # Use 90th percentile of valid gradients for sharp, distinct fronts
        valid_grads = grad[valid]
        if len(valid_grads) > 0:
            threshold = np.percentile(valid_grads, self.front_percentile)
        else:
            threshold = 0
            
        front_mask = (grad > threshold) & valid
        return front_mask

    def _detect_eddies(self, uo_orig: np.ndarray, vo_orig: np.ndarray, ssh_orig: np.ndarray, 
                       lat_vals: np.ndarray, lon_vals: np.ndarray) -> List[Dict[str, Any]]:
        """Optimized Eddy Detection with SSH cross-verification."""
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

        # Standard Oceanographic Threshold: W < -0.2 * variance(W)
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
            if pixels < self.min_eddy_pixels: continue
            
            rows, cols = np.where(blob)
            r, c = int(rows.mean()), int(cols.mean())
            radius = float(np.sqrt((pixels * pixel_area) / np.pi))
            
            vort_at_center = vort[r, c]
            is_cyclonic = vort_at_center > 0
            eddy_type = "cyclonic_cold_core" if is_cyclonic else "anticyclonic_warm_core"
            
            # SSH Cross-Verification (Cyclonic = negative SSH, Anticyclonic = positive SSH)
            ssh_val = None
            if ssh_orig is not None and r < ssh_orig.shape[0] and c < ssh_orig.shape[1]:
                val = ssh_orig[r, c]
                if np.isfinite(val):
                    ssh_val = float(val)
                    # Discard if physics don't match (Fake eddy)
                    if is_cyclonic and ssh_val > 0: continue
                    if not is_cyclonic and ssh_val < 0: continue

            strength = float(np.nanmean(np.abs(W[blob])))

            eddies.append({
                "center_lat": float(lat_vals[r]), "center_lon": float(lon_vals[c]),
                "radius_km": round(radius, 2), "type": eddy_type, 
                "strength": round(strength, 4), "ssh_at_center": ssh_val, 
                "pixel_count": pixels
            })
        return eddies

    def _format_output(self, front_mask: np.ndarray, eddies: List[Dict[str, Any]], 
                       lat_vals: np.ndarray, lon_vals: np.ndarray) -> Dict[str, Any]:
        lons_grid, lats_grid = np.meshgrid(lon_vals, lat_vals)
        front_lats = lats_grid[front_mask].tolist()
        front_lons = lons_grid[front_mask].tolist()
        fronts = [{"lat": lat, "lon": lon} for lat, lon in zip(front_lats, front_lons)]
        return {"fronts": fronts, "eddies": eddies}

    def predict(self, daily_ds: xr.Dataset) -> Dict[str, Any]:
        """
        Main API Method. Pass an xarray Dataset for a single day, get JSON output.
        """
        sst, ssh, uo, vo, lats, lons = self._preprocess(daily_ds)
        front_mask = self._detect_fronts(sst)
        eddies = self._detect_eddies(uo, vo, ssh, lats, lons)
        return self._format_output(front_mask, eddies, lats, lons)