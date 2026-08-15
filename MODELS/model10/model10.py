# model10_service.py
import numpy as np
import math
from typing import Dict, List, Any
from collections import defaultdict

class CollisionDetectionModel:
    """
    BlueFish AI - Model 10: Vessel Collision & Near-Miss Detector
    Uses spatial hashing + CPA/TCPA kinematics for O(N) scalability.
    """
    
    def __init__(self, cpa_threshold_km: float = 0.5, tcpa_threshold_min: float = 15.0, grid_size_km: float = 20.0):
        self.cpa_threshold_km = cpa_threshold_km
        self.tcpa_threshold_min = tcpa_threshold_min
        self.grid_size_km = grid_size_km
        self.earth_radius_km = 6371.0

    def _to_xy_km(self, lat, lon, ref_lat, ref_lon):
        """Converts Lat/Lon to flat X/Y in kilometers relative to a reference point."""
        x = lon * (math.pi / 180.0) * self.earth_radius_km * math.cos(math.radians(ref_lat))
        y = lat * (math.pi / 180.0) * self.earth_radius_km
        ref_x = ref_lon * (math.pi / 180.0) * self.earth_radius_km * math.cos(math.radians(ref_lat))
        ref_y = ref_lat * (math.pi / 180.0) * self.earth_radius_km
        return x - ref_x, y - ref_y

    def _calculate_cpa_tcpa(self, v1: Dict, v2: Dict) -> tuple:
        """Calculates Closest Point of Approach (CPA) and Time to CPA (TCPA)."""
        ref_lat, ref_lon = v1['lat'], v1['lon']
        x2, y2 = self._to_xy_km(v2['lat'], v2['lon'], ref_lat, ref_lon)
        
        # Convert speed (knots) to km/min
        spd1_kmm = v1['speed'] * 1.852 / 60.0
        spd2_kmm = v2['speed'] * 1.852 / 60.0
        
        # Velocity vectors (X=East, Y=North)
        vx1 = spd1_kmm * math.sin(math.radians(v1['heading']))
        vy1 = spd1_kmm * math.cos(math.radians(v1['heading']))
        vx2 = spd2_kmm * math.sin(math.radians(v2['heading']))
        vy2 = spd2_kmm * math.cos(math.radians(v2['heading']))
        
        # Relative position and velocity
        dx = x2
        dy = y2
        dvx = vx2 - vx1
        dvy = vy2 - vy1
        
        rel_speed_sq = dvx**2 + dvy**2
        if rel_speed_sq < 1e-6:
            tcpa = 0.0
        else:
            tcpa = -(dx * dvx + dy * dvy) / rel_speed_sq
            
        if tcpa <= 0:
            cpa_dist = math.sqrt(dx**2 + dy**2)
            tcpa = 0.0
        else:
            cpa_x = dx + dvx * tcpa
            cpa_y = dy + dvy * tcpa
            cpa_dist = math.sqrt(cpa_x**2 + cpa_y**2)
            
        return cpa_dist, tcpa

    def predict(self, vessels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main API Method. Pass a list of active vessels, get collision alerts.
        Required keys: 'mmsi', 'lat', 'lon', 'speed', 'heading'
        """
        if len(vessels) < 2:
            return {"alerts": [], "vessels_checked": len(vessels)}
            
        alerts = []
        
        # 1. Build Spatial Hash Grid (O(N) complexity)
        # Groups boats into 20km cells so we don't compare boats 1000km apart
        grid = defaultdict(list)
        for v in vessels:
            # Convert lat/lon to rough grid indices
            grid_lat = int(v['lat'] * (111.0 / self.grid_size_km))
            grid_lon = int(v['lon'] * (111.0 / self.grid_size_km))
            grid[(grid_lat, grid_lon)].append(v)
            
        # 2. Check pairs only within the same or adjacent grids
        checked_pairs = set()
        
        for (glat, glon), cell_vessels in grid.items():
            # Get vessels from this cell and the 8 adjacent cells
            nearby_vessels = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    nearby_vessels.extend(grid.get((glat + i, glon + j), []))
                    
            for v1 in cell_vessels:
                for v2 in nearby_vessels:
                    if v1['mmsi'] == v2['mmsi']: continue
                    
                    # Prevent checking the same pair twice
                    pair_id = tuple(sorted((v1['mmsi'], v2['mmsi'])))
                    if pair_id in checked_pairs: continue
                    checked_pairs.add(pair_id)
                    
                    cpa, tcpa = self._calculate_cpa_tcpa(v1, v2)
                    
                    if cpa <= self.cpa_threshold_km and tcpa <= self.tcpa_threshold_min and tcpa >= 0:
                        # Calculate exact collision Lat/Lon
                        collision_lat = v1['lat'] + (v1['speed'] * 1.852 / 60.0 * math.cos(math.radians(v1['heading'])) * tcpa) / 111.0
                        collision_lon = v1['lon'] + (v1['speed'] * 1.852 / 60.0 * math.sin(math.radians(v1['heading'])) * tcpa) / (111.0 * math.cos(math.radians(v1['lat'])))
                        
                        alerts.append({
                            "vessel_1_mmsi": v1['mmsi'], "vessel_2_mmsi": v2['mmsi'],
                            "cpa_km": round(cpa, 3), "tcpa_min": round(tcpa, 1),
                            "collision_lat": round(collision_lat, 4), 
                            "collision_lon": round(collision_lon, 4),
                            "severity": "HIGH" if cpa < 0.2 else "MEDIUM"
                        })
                    
        return {"alerts": alerts, "vessels_checked": len(vessels)}