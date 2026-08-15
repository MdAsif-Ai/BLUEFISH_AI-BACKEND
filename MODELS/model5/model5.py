# model5_service.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from typing import Dict, List, Any

class FleetDensityModel:
    """
    BlueFish AI - Model 5: Fleet Density & Overcrowding Detector
    
    A production-grade estimator class that wraps DBSCAN spatial clustering.
    No training required. Call .predict() to get JSON-ready map data.
    """
    
    def __init__(self, eps_km: float = 5.0, min_vessels: int = 10,
                 lat_min: float = 6.0, lat_max: float = 23.0, 
                 lon_min: float = 68.0, lon_max: float = 89.0):
        """
        Initialize the model with geographic bounds and clustering parameters.
        
        Args:
            eps_km: The maximum distance (in km) between two boats for them to be 
                    considered in the same overcrowded zone.
            min_vessels: The minimum number of boats required to form a crowded zone.
        """
        self.eps_km = eps_km
        self.min_vessels = min_vessels
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.earth_radius_km = 6371.0088
        
        # The underlying scikit-learn model (instantiated during predict)
        self._model = None

    def _preprocess(self, df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Filters raw AIS data to the target date and region without mutating the original df."""
        # Work on a copy to prevent SettingWithCopyWarning in production APIs
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        target_dt = pd.to_datetime(target_date).normalize()
        
        mask = (
            (df['date'].dt.normalize() == target_dt) &
            (df['cell_ll_lat'].between(self.lat_min, self.lat_max)) &
            (df['cell_ll_lon'].between(self.lon_min, self.lon_max)) &
            (df['fishing_hours'] > 0)
        )
        day_df = df[mask].copy()

        if day_df.empty:
            return day_df

        # Deduplicate by MMSI (Keep the row with max fishing_hours)
        if 'mmsi' in day_df.columns:
            idx_max = day_df.groupby('mmsi')['fishing_hours'].idxmax()
            day_df = day_df.loc[idx_max].copy()

        return day_df

    def _cluster(self, day_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the DBSCAN clustering algorithm on the coordinates."""
        if day_df.empty or len(day_df) < self.min_vessels:
            day_df['cluster_id'] = -1
            return day_df

        # Vectorized NumPy conversion for speed
        coords_rad = np.radians(day_df[['cell_ll_lat', 'cell_ll_lon']].to_numpy())
        eps_rad = self.eps_km / self.earth_radius_km

        # Initialize and run DBSCAN
        self._model = DBSCAN(
            eps=eps_rad, 
            min_samples=self.min_vessels, 
            metric="haversine", 
            algorithm="ball_tree"
        )
        
        day_df['cluster_id'] = self._model.fit_predict(coords_rad)
        return day_df

    def _format_output(self, day_df: pd.DataFrame) -> Dict[str, Any]:
        """Formats the clustered dataframe into JSON-ready dictionaries."""
        if day_df.empty:
            return {"zones": [], "boats": []}

        clusters = day_df[day_df['cluster_id'] != -1].copy()
        zones = []

        for cid in clusters['cluster_id'].unique():
            c = clusters[clusters['cluster_id'] == cid]
            size = len(c)
            
            center_lat = c['cell_ll_lat'].mean()
            center_lon = c['cell_ll_lon'].mean()

            # Calculate bounding box area in km
            lat_km = (c['cell_ll_lat'].max() - c['cell_ll_lat'].min()) * 111.0
            lon_km = (c['cell_ll_lon'].max() - c['cell_ll_lon'].min()) * 111.0 * np.cos(np.radians(center_lat))
            area = max(lat_km * lon_km, 0.1)  # Prevent division by zero
            density = size / area

            # Determine severity based on density
            if density > 5:
                severity = "SEVERE"
            elif density > 2:
                severity = "HIGH"
            else:
                severity = "MODERATE"

            zones.append({
                "cluster_id": int(cid),
                "vessels": int(size),
                "center_lat": float(center_lat),
                "center_lon": float(center_lon),
                "area_km2": round(float(area), 2),
                "density": round(float(density), 2),
                "severity": severity
            })

        # Sort zones by density (most crowded first)
        zones.sort(key=lambda x: x['density'], reverse=True)

        # Format individual boat points
        boats = day_df[['cell_ll_lat', 'cell_ll_lon', 'cluster_id']].to_dict('records')
        for b in boats:
            b['lat'] = b.pop('cell_ll_lat')
            b['lon'] = b.pop('cell_ll_lon')

        return {"zones": zones, "boats": boats}

    def predict(self, df: pd.DataFrame, target_date: str) -> Dict[str, Any]:
        """
        Main API Method. Pass raw dataframe and target date, get JSON output.
        
        Example:
            model = FleetDensityModel()
            result = model.predict(vessel_df, "2023-10-10")
        """
        # 1. Clean and filter data
        day_df = self._preprocess(df, target_date)
        
        # 2. Run clustering math
        clustered_df = self._cluster(day_df)
        
        # 3. Return formatted JSON
        return self._format_output(clustered_df)

