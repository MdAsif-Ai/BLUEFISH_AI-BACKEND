"""
BlueFish AI - Model 8 Service: Solunar Feeding Time Windows
============================================================
Wraps the solunar astronomical math model.

Solunar theory: fish feeding activity peaks at specific times related
to the sun and moon's positions (transits, opposition, new/full moon).
This model is PURE MATH — no ML artifacts, no ocean data needed.

Feeding window classification:
  - MAJOR (2h): Moon transit (overhead) or opposition (underfoot)
  - MINOR (1h): Sunrise or sunset
  - Daily rating (1–5): Based on moon phase proximity to new/full moon

Usage:
    service = TimeWindowService(model_registry.model8)
    result = service.predict("2024-12-15", 10.8, 79.8)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bluefish.services.model8")


class TimeWindowModel:
    """
    BlueFish AI - Model 8: Optimal Time-Window Recommendation
    Uses Solunar Theory (Sun/Moon transit) + Moon Phase to predict peak feeding times.
    """

    def __init__(self, major_window_mins: int = 120, minor_window_mins: int = 60):
        self.major_duration = timedelta(minutes=major_window_mins)
        self.minor_duration = timedelta(minutes=minor_window_mins)

    def _calculate_sun_events(self, date: datetime, lat: float, lon: float) -> Tuple[datetime, datetime, datetime]:
        N = date.timetuple().tm_yday
        decl = 23.45 * math.sin(math.radians(360 * (284 + N) / 365))
        try:
            ha = math.degrees(math.acos(
                math.cos(math.radians(90.833)) /
                (math.cos(math.radians(lat)) * math.cos(math.radians(decl)))
                - math.tan(math.radians(lat)) * math.tan(math.radians(decl))
            ))
        except ValueError:
            ha = 90.0

        solar_noon = 12.0 - (lon / 15.0)
        sunrise = max(0.0, min(23.99, solar_noon - (ha / 15.0)))
        sunset = max(0.0, min(23.99, solar_noon + (ha / 15.0)))
        solar_noon_clean = max(0.0, min(23.99, solar_noon))

        sr_dt = date.replace(hour=int(sunrise), minute=int((sunrise % 1) * 60), second=0)
        ss_dt = date.replace(hour=int(sunset), minute=int((sunset % 1) * 60), second=0)
        noon_dt = date.replace(hour=int(solar_noon_clean), minute=int((solar_noon_clean % 1) * 60), second=0)

        return sr_dt, ss_dt, noon_dt

    def _calculate_moon_events(self, date: datetime) -> Tuple[datetime, datetime]:
        lunar_offset = 50.0 / 60.0
        moon_transit = date.replace(hour=12, minute=0) + timedelta(hours=lunar_offset / 2)
        moon_antitransit = moon_transit - timedelta(hours=12.4)
        return moon_transit, moon_antitransit

    def _calculate_moon_phase_and_rating(self, date: datetime) -> Tuple[int, str, float]:
        """Calculates moon phase (0=New, 0.5=Full, 1=New) and a 1-5 star rating."""
        ref_date = datetime(2000, 1, 6)
        days_since = (date - ref_date).days
        phase = (days_since % 29.53) / 29.53  # 0 to 1

        phase_factor = abs(phase - 0.5)
        if phase_factor < 0.05 or phase_factor > 0.45:
            rating = 5
        elif phase_factor < 0.15 or phase_factor > 0.35:
            rating = 4
        elif phase_factor < 0.25:
            rating = 3
        else:
            rating = 2

        if phase < 0.03 or phase > 0.97:
            phase_name = "New Moon"
        elif phase < 0.47 or phase > 0.53:
            phase_name = "Full Moon"
        elif phase < 0.25:
            phase_name = "First Quarter"
        elif phase < 0.75:
            phase_name = "Last Quarter"
        else:
            phase_name = "Waxing/Waning"

        phase_pct = round(phase * 100.0, 1)
        return rating, phase_name, phase_pct

    def predict(self, target_date_str: str, lat: float, lon: float) -> Dict[str, Any]:
        try:
            date_obj = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.now()

        sunrise, sunset, solar_noon = self._calculate_sun_events(date_obj, lat, lon)
        moon_transit, moon_antitransit = self._calculate_moon_events(date_obj)
        rating, phase_name, phase_pct = self._calculate_moon_phase_and_rating(date_obj)

        windows = []

        # Major 1: Moon Overhead
        m1_start = moon_transit - (self.major_duration / 2)
        m1_end = moon_transit + (self.major_duration / 2)
        windows.append({
            "type": "MAJOR",
            "start_time": m1_start.strftime("%H:%M"),
            "end_time": m1_end.strftime("%H:%M"),
            "peak_time": moon_transit.strftime("%H:%M"),
            "duration_minutes": 120,
            "quality": "EXCELLENT" if rating >= 4 else "GOOD",
            "reason": "Moon Overhead",
        })

        # Major 2: Moon Underfoot
        m2_start = moon_antitransit - (self.major_duration / 2)
        m2_end = moon_antitransit + (self.major_duration / 2)
        windows.append({
            "type": "MAJOR",
            "start_time": m2_start.strftime("%H:%M"),
            "end_time": m2_end.strftime("%H:%M"),
            "peak_time": moon_antitransit.strftime("%H:%M"),
            "duration_minutes": 120,
            "quality": "EXCELLENT" if rating >= 4 else "GOOD",
            "reason": "Moon Underfoot",
        })

        # Minor 1: Sunrise
        s1_start = sunrise - (self.minor_duration / 2)
        s1_end = sunrise + (self.minor_duration / 2)
        windows.append({
            "type": "MINOR",
            "start_time": s1_start.strftime("%H:%M"),
            "end_time": s1_end.strftime("%H:%M"),
            "peak_time": sunrise.strftime("%H:%M"),
            "duration_minutes": 60,
            "quality": "GOOD" if rating >= 3 else "FAIR",
            "reason": "Sunrise",
        })

        # Minor 2: Sunset
        s2_start = sunset - (self.minor_duration / 2)
        s2_end = sunset + (self.minor_duration / 2)
        windows.append({
            "type": "MINOR",
            "start_time": s2_start.strftime("%H:%M"),
            "end_time": s2_end.strftime("%H:%M"),
            "peak_time": sunset.strftime("%H:%M"),
            "duration_minutes": 60,
            "quality": "GOOD" if rating >= 3 else "FAIR",
            "reason": "Sunset",
        })

        windows.sort(key=lambda x: x["start_time"])

        return {
            "date": target_date_str,
            "location": {"lat": lat, "lon": lon},
            "daily_rating": rating,
            "moon_phase": phase_name,
            "moon_phase_pct": phase_pct,
            "sunrise": sunrise.strftime("%H:%M"),
            "sunset": sunset.strftime("%H:%M"),
            "feeding_windows": windows,
        }


class TimeWindowService:
    """
    Solunar time window prediction service.
    Wraps TimeWindowModel. Always available — pure astronomical math.
    """

    def __init__(self, model8_instance: Optional[Any] = None):
        if model8_instance is None:
            self.model = TimeWindowModel()
        else:
            self.model = model8_instance

    def predict(
        self,
        date_str: str,
        lat: float,
        lon: float,
    ) -> Dict[str, Any]:
        """
        Calculates solunar feeding time windows for the given date and location.
        """
        try:
            result = self.model.predict(date_str, lat, lon)
            return _normalize_time_window_result(result, date_str, lat, lon)
        except Exception as e:
            logger.error(f"Time window prediction failed for {date_str} ({lat},{lon}): {e}")
            return _fallback_response(date_str, lat, lon, str(e))

    def predict_range(
        self,
        start_date: str,
        end_date: str,
        lat: float,
        lon: float,
    ) -> List[Dict[str, Any]]:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Invalid date range: {e}")
            return []

        results = []
        current = start
        while current <= end:
            ds = current.strftime("%Y-%m-%d")
            results.append(self.predict(ds, lat, lon))
            current += timedelta(days=1)

        return results

    def best_days_in_month(
        self,
        year: int,
        month: int,
        lat: float,
        lon: float,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        import calendar

        _, days_in_month = calendar.monthrange(year, month)
        all_days = []
        for day in range(1, days_in_month + 1):
            ds = f"{year:04d}-{month:02d}-{day:02d}"
            result = self.predict(ds, lat, lon)
            all_days.append(result)

        all_days.sort(key=lambda x: x.get("daily_rating", 0), reverse=True)
        return all_days[:top_n]


def _normalize_time_window_result(
    raw: Dict[str, Any],
    date_str: str,
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    return {
        "date": raw.get("date", date_str),
        "location": {"lat": lat, "lon": lon},
        "daily_rating": raw.get("daily_rating", 3),
        "moon_phase": raw.get("moon_phase", "unknown"),
        "moon_phase_pct": raw.get("moon_phase_pct", 50.0),
        "sunrise": raw.get("sunrise", "06:00"),
        "sunset": raw.get("sunset", "18:30"),
        "moonrise": raw.get("moonrise", "unknown"),
        "moonset": raw.get("moonset", "unknown"),
        "feeding_windows": raw.get("feeding_windows", []),
    }


def _fallback_response(date_str: str, lat: float, lon: float, error: str) -> Dict[str, Any]:
    return {
        "date": date_str,
        "location": {"lat": lat, "lon": lon},
        "daily_rating": 3,
        "moon_phase": "unknown",
        "moon_phase_pct": 50.0,
        "sunrise": "06:00",
        "sunset": "18:30",
        "moonrise": "unknown",
        "moonset": "unknown",
        "feeding_windows": [
            {
                "type": "MINOR",
                "start_time": "06:00",
                "end_time": "07:00",
                "duration_minutes": 60,
                "quality": "FAIR",
                "reason": "Sunrise",
            }
        ],
        "degraded": True,
        "degraded_reason": error,
    }
