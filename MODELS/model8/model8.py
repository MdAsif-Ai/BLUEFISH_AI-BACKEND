# model8_service.py
import math
from typing import Dict, List, Any
from datetime import datetime, timedelta

class TimeWindowModel:
    """
    BlueFish AI - Model 8: Optimal Time-Window Recommendation
    Uses Solunar Theory (Sun/Moon transit) + Moon Phase to predict peak feeding times.
    """
    
    def __init__(self, major_window_mins: int = 120, minor_window_mins: int = 60):
        self.major_duration = timedelta(minutes=major_window_mins)
        self.minor_duration = timedelta(minutes=minor_window_mins)

    def _calculate_sun_events(self, date: datetime, lat: float, lon: float) -> tuple:
        N = date.timetuple().tm_yday
        decl = 23.45 * math.sin(math.radians(360 * (284 + N) / 365))
        try:
            ha = math.degrees(math.acos(
                math.cos(math.radians(90.833)) / 
                (math.cos(math.radians(lat)) * math.cos(math.radians(decl))) 
                - math.tan(math.radians(lat)) * math.tan(math.radians(decl))
            ))
        except ValueError:
            return 6.0, 18.0
            
        solar_noon = 12.0 - (lon / 15.0)
        sunrise = solar_noon - (ha / 15.0)
        sunset = solar_noon + (ha / 15.0)
        
        sr_dt = date.replace(hour=int(sunrise), minute=int((sunrise % 1) * 60), second=0)
        ss_dt = date.replace(hour=int(sunset), minute=int((sunset % 1) * 60), second=0)
        noon_dt = date.replace(hour=int(solar_noon), minute=int((solar_noon % 1) * 60), second=0)
        
        return sr_dt, ss_dt, noon_dt

    def _calculate_moon_events(self, date: datetime) -> tuple:
        lunar_offset = 50.0 / 60.0
        moon_transit = date.replace(hour=12, minute=0) + timedelta(hours=lunar_offset/2)
        moon_antitransit = moon_transit - timedelta(hours=12.4)
        return moon_transit, moon_antitransit

    def _calculate_moon_phase_and_rating(self, date: datetime) -> tuple:
        """Calculates moon phase (0=New, 0.5=Full, 1=New) and a 1-5 star rating."""
        # Known reference New Moon date (Jan 6, 2000)
        ref_date = datetime(2000, 1, 6)
        days_since = (date - ref_date).days
        phase = (days_since % 29.53) / 29.53  # 0 to 1
        
        # Rating is highest near New Moon (0) and Full Moon (0.5)
        phase_factor = abs(phase - 0.5)
        if phase_factor < 0.05 or phase_factor > 0.45: # New or Full Moon
            rating = 5
        elif phase_factor < 0.15 or phase_factor > 0.35:
            rating = 4
        elif phase_factor < 0.25:
            rating = 3
        else:
            rating = 2
            
        if phase < 0.03 or phase > 0.97: phase_name = "New Moon"
        elif phase < 0.47 or phase > 0.53: phase_name = "Full Moon"
        elif phase < 0.25: phase_name = "First Quarter"
        elif phase < 0.75: phase_name = "Last Quarter"
        else: phase_name = "Waxing/Waning"
        
        return rating, phase_name

    def predict(self, target_date_str: str, lat: float, lon: float) -> Dict[str, Any]:
        date_obj = datetime.strptime(target_date_str, "%Y-%m-%d")
        
        sunrise, sunset, solar_noon = self._calculate_sun_events(date_obj, lat, lon)
        moon_transit, moon_antitransit = self._calculate_moon_events(date_obj)
        rating, phase_name = self._calculate_moon_phase_and_rating(date_obj)
        
        windows = []
        
        # Major 1: Moon Overhead
        start = moon_transit - (self.major_duration / 2)
        end = moon_transit + (self.major_duration / 2)
        windows.append({"type": "MAJOR", "start_time": start.strftime("%H:%M"), "end_time": end.strftime("%H:%M"), "peak_time": moon_transit.strftime("%H:%M"), "reason": "Moon Overhead"})
        
        # Major 2: Moon Underfoot
        start = moon_antitransit - (self.major_duration / 2)
        end = moon_antitransit + (self.major_duration / 2)
        windows.append({"type": "MAJOR", "start_time": start.strftime("%H:%M"), "end_time": end.strftime("%H:%M"), "peak_time": moon_antitransit.strftime("%H:%M"), "reason": "Moon Underfoot"})
        
        # Minor 1: Sunrise
        start = sunrise - (self.minor_duration / 2)
        end = sunrise + (self.minor_duration / 2)
        windows.append({"type": "MINOR", "start_time": start.strftime("%H:%M"), "end_time": end.strftime("%H:%M"), "peak_time": sunrise.strftime("%H:%M"), "reason": "Sunrise"})
        
        # Minor 2: Sunset
        start = sunset - (self.minor_duration / 2)
        end = sunset + (self.minor_duration / 2)
        windows.append({"type": "MINOR", "start_time": start.strftime("%H:%M"), "end_time": end.strftime("%H:%M"), "peak_time": sunset.strftime("%H:%M"), "reason": "Sunset"})
        
        windows.sort(key=lambda x: x['start_time'])
        
        return {
            "date": target_date_str,
            "location": {"lat": lat, "lon": lon},
            "daily_rating": rating, # 1 to 5 Stars
            "moon_phase": phase_name,
            "sunrise": sunrise.strftime("%H:%M"),
            "sunset": sunset.strftime("%H:%M"),
            "feeding_windows": windows
        }