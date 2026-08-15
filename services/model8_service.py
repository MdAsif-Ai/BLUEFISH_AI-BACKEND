"""
BlueFish AI - Model 8 Service: Solunar Feeding Time Windows
============================================================
Wraps the solunar astronomical math model (model8.py).

Solunar theory: fish feeding activity peaks at specific times related
to the sun and moon's positions (transits, opposition, new/full moon).
This model is PURE MATH — no ML artifacts, no ocean data needed.
It's always available regardless of model loading failures.

Feeding window classification:
  - MAJOR (2h): Moon transit (overhead) or opposition (underfoot)
  - MINOR (1h): Moon rising or setting
  - Daily rating (1–5): Based on moon phase proximity to new/full

Usage:
    service = TimeWindowService(model_registry.model8)
    result = service.predict("2024-12-15", 10.8, 79.8)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bluefish.services.model8")


class TimeWindowService:
    """
    Solunar time window prediction service.
    Wraps TimeWindowModel from MODELS/model8/model8.py.
    Always available — pure astronomical math.
    """

    def __init__(self, model8_instance):
        self.model = model8_instance

    def predict(
        self,
        date_str: str,
        lat: float,
        lon: float,
    ) -> Dict[str, Any]:
        """
        Calculates solunar feeding time windows for the given date and location.

        Args:
            date_str: Date in YYYY-MM-DD format
            lat: Latitude of the fishing location
            lon: Longitude of the fishing location

        Returns:
            {
                "date": str,
                "location": {lat, lon},
                "daily_rating": int (1-5, 5 = best fishing day),
                "moon_phase": str,
                "moon_phase_pct": float (0=new, 50=half, 100=full),
                "sunrise": str (HH:MM local IST),
                "sunset": str (HH:MM local IST),
                "moonrise": str,
                "moonset": str,
                "feeding_windows": [
                    {
                        "start": str (HH:MM IST),
                        "end": str (HH:MM IST),
                        "duration_minutes": int,
                        "type": "MAJOR" | "MINOR",
                        "quality": "EXCELLENT" | "GOOD" | "FAIR",
                    }
                ],
            }
        """
        try:
            result = self.model.predict(date_str, lat, lon)
            # Normalize result format in case model8 returns different key names
            return _normalize_time_window_result(result, date_str, lat, lon)
        except Exception as e:
            logger.error(f"Time window prediction failed for {date_str} ({lat},{lon}): {e}")
            # Return a graceful degraded response rather than 500
            return _fallback_response(date_str, lat, lon, str(e))

    def predict_range(
        self,
        start_date: str,
        end_date: str,
        lat: float,
        lon: float,
    ) -> List[Dict[str, Any]]:
        """
        Predicts feeding windows for a date range.
        Useful for the mobile app's weekly fishing calendar.
        """
        from datetime import date, timedelta

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
        """
        Returns the top N best fishing days in a given month based on solunar rating.
        Used by the Flutter app's monthly planner calendar.
        """
        import calendar

        _, days_in_month = calendar.monthrange(year, month)
        all_days = []
        for day in range(1, days_in_month + 1):
            ds = f"{year:04d}-{month:02d}-{day:02d}"
            result = self.predict(ds, lat, lon)
            all_days.append(result)

        # Sort by daily_rating descending
        all_days.sort(key=lambda x: x.get("daily_rating", 0), reverse=True)
        return all_days[:top_n]


def _normalize_time_window_result(
    raw: Dict[str, Any],
    date_str: str,
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    """Normalizes the raw model8 output to our standard schema."""
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
    """Returns a degraded response when model8 fails."""
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
                "start": "06:00",
                "end": "07:00",
                "duration_minutes": 60,
                "type": "MINOR",
                "quality": "FAIR",
                "note": "Degraded — solunar model unavailable",
            }
        ],
        "degraded": True,
        "degraded_reason": error,
    }
