# model7_service.py
import numpy as np
import xarray as xr
import heapq
from typing import List, Tuple, Dict

class RouteOptimizationModel:
    """
    BlueFish AI - Model 7: Fuel-Efficient Route Optimization
    Uses A* Pathfinding weighted by ocean current resistance and bathymetry.
    """
    
    def __init__(self, cost_per_km: float = 1.0, cost_against_current: float = 10.0, 
                 cost_with_current: float = 0.1, min_depth_m: float = -5.0):
        self.cost_per_km = cost_per_km
        self.cost_against_current = cost_against_current
        self.cost_with_current = cost_with_current
        self.min_depth_m = min_depth_m
        self.grid_lat = None
        self.grid_lon = None
        self.uo = None
        self.vo = None
        self.depth = None

    def load_environment(self, currents_ds: xr.Dataset, bathymetry_ds: xr.Dataset):
        """Loads daily currents and static bathymetry into memory."""
        # Extract currents
        uo_da = currents_ds['uo']
        vo_da = currents_ds['vo']
        
        if 'depth' in uo_da.dims: uo_da = uo_da.isel(depth=0)
        if 'depth' in vo_da.dims: vo_da = vo_da.isel(depth=0)
            
        self.grid_lat = uo_da['latitude'].values if 'latitude' in uo_da.coords else uo_da['lat'].values
        self.grid_lon = uo_da['longitude'].values if 'longitude' in uo_da.coords else uo_da['lon'].values
        
        self.uo = np.nan_to_num(uo_da.values, nan=0.0)
        self.vo = np.nan_to_num(vo_da.values, nan=0.0)
        
        # Interpolate Bathymetry to match Currents grid
        bath_da = bathymetry_ds['elevation']
        bath_lat = bath_da['lat'].values if 'lat' in bath_da.coords else bath_da['latitude'].values
        bath_lon = bath_da['lon'].values if 'lon' in bath_da.coords else bath_da['longitude'].values
        
        # Simple nearest-index mapping for speed in production
        # Find the bathymetry indices closest to the currents grid
        lat_idx = np.searchsorted(bath_lat, self.grid_lat)
        lon_idx = np.searchsorted(bath_lon, self.grid_lon)
        
        # Clip to valid bounds
        lat_idx = np.clip(lat_idx, 0, len(bath_lat)-1)
        lon_idx = np.clip(lon_idx, 0, len(bath_lon)-1)
        
        self.depth = bath_da.values[np.ix_(lat_idx, lon_idx)]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _get_cost(self, current: Tuple[int, int], nxt: Tuple[int, int]) -> float:
        r1, c1 = current
        r2, c2 = nxt
        
        # 1. Bathymetry check (Cannot drive on land < 5m deep)
        if self.depth[r2, c2] > self.min_depth_m:
            return float('inf')
            
        # 2. Base distance cost (in grid steps)
        distance = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
        
        # 3. Current resistance cost (Vector Math)
        travel_dir = np.array([r2 - r1, c2 - c1])
        norm = np.linalg.norm(travel_dir)
        if norm == 0: return float('inf')
        travel_dir = travel_dir / norm
        
        current_vec = np.array([self.uo[r2, c2], self.vo[r2, c2]])
        
        # Dot product: 1 = perfectly with current, -1 = perfectly against
        alignment = np.dot(travel_dir, current_vec)
        
        if alignment > 0:
            current_cost = self.cost_with_current * (1 - alignment)
        else:
            current_cost = self.cost_against_current * (-alignment)
            
        return distance * self.cost_per_km + current_cost

    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Removes jagged grid steps to create a smooth navigational line."""
        if len(path) < 3: return path
        smoothed = [path[0]]
        for i in range(1, len(path)-1):
            # Only keep points where the direction changes
            prev_dir = (path[i][0] - smoothed[-1][0], path[i][1] - smoothed[-1][1])
            next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if prev_dir != next_dir:
                smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

    def predict(self, start_lat: float, start_lon: float, target_lat: float, target_lon: float) -> Dict:
        """Calculates the optimal fuel-efficient route."""
        start_r = np.argmin(np.abs(self.grid_lat - start_lat))
        start_c = np.argmin(np.abs(self.grid_lon - start_lon))
        target_r = np.argmin(np.abs(self.grid_lat - target_lat))
        target_c = np.argmin(np.abs(self.grid_lon - target_lon))
        
        start_node = (int(start_r), int(start_c))
        target_node = (int(target_r), int(target_c))
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, target_node)}
        
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == target_node:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                path = self._smooth_path(path)
                route_coords = [{"lat": float(self.grid_lat[r]), "lon": float(self.grid_lon[c])} for r, c in path]
                return {"route": route_coords, "steps": len(route_coords)}
                
            visited.add(current)
            
            r, c = current
            # 8 neighbors (N, S, E, W, Diagonals)
            neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1), (r-1,c-1), (r-1,c+1), (r+1,c-1), (r+1,c+1)]
            
            for neighbor in neighbors:
                nr, nc = neighbor
                if 0 <= nr < self.uo.shape[0] and 0 <= nc < self.uo.shape[1]:
                    if neighbor in visited: continue
                        
                    cost = self._get_cost(current, neighbor)
                    if cost == float('inf'): continue
                        
                    tentative_g_score = g_score[current] + cost
                    
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, target_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return {"route": [], "steps": 0}