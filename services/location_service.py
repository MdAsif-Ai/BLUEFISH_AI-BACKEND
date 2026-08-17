"""
BlueFish AI - Universal Marine Location Geocoding Service
===========================================================
Queries Open-Meteo Geocoding API for any harbour, port, coastal city, or marine coordinate.
Enriches results with known Tamil Nadu / Indian EEZ fishing harbours for exact maritime matching.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List
import httpx

logger = logging.getLogger("bluefish.services.location")

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Curated maritime & fishing harbour database overlay for high-precision maritime queries
MARITIME_HARBOUR_OVERLAY: List[Dict[str, Any]] = [
    {
        "id": 9001,
        "name": "Kasimedu Fishing Harbour",
        "region": "Chennai, Tamil Nadu",
        "country": "India",
        "latitude": 13.1167,
        "longitude": 80.2833,
        "timezone": "Asia/Kolkata",
        "display_name": "Kasimedu Fishing Harbour, Chennai, Tamil Nadu, India",
    },
    {
        "id": 9002,
        "name": "Cuddalore Old Town Port",
        "region": "Cuddalore, Tamil Nadu",
        "country": "India",
        "latitude": 11.7500,
        "longitude": 79.7700,
        "timezone": "Asia/Kolkata",
        "display_name": "Cuddalore Old Town Port, Cuddalore, Tamil Nadu, India",
    },
    {
        "id": 9003,
        "name": "Nagapattinam Port",
        "region": "Nagapattinam, Tamil Nadu",
        "country": "India",
        "latitude": 10.7656,
        "longitude": 79.8428,
        "timezone": "Asia/Kolkata",
        "display_name": "Nagapattinam Port, Nagapattinam, Tamil Nadu, India",
    },
    {
        "id": 9004,
        "name": "Pamban Fishing Harbour",
        "region": "Ramanathapuram, Tamil Nadu",
        "country": "India",
        "latitude": 9.2800,
        "longitude": 79.3000,
        "timezone": "Asia/Kolkata",
        "display_name": "Pamban Fishing Harbour, Ramanathapuram, Tamil Nadu, India",
    },
    {
        "id": 9005,
        "name": "Thoothukudi Fishing Harbour",
        "region": "Thoothukudi, Tamil Nadu",
        "country": "India",
        "latitude": 8.7600,
        "longitude": 78.1300,
        "timezone": "Asia/Kolkata",
        "display_name": "Thoothukudi Fishing Harbour, Thoothukudi, Tamil Nadu, India",
    },
    {
        "id": 9006,
        "name": "Colachel Fishing Harbour",
        "region": "Kanyakumari, Tamil Nadu",
        "country": "India",
        "latitude": 8.1750,
        "longitude": 77.3080,
        "timezone": "Asia/Kolkata",
        "display_name": "Colachel Fishing Harbour, Kanyakumari, Tamil Nadu, India",
    },
    {
        "id": 9007,
        "name": "Sassoon Dock Fishing Harbour",
        "region": "Mumbai, Maharashtra",
        "country": "India",
        "latitude": 18.9100,
        "longitude": 72.8250,
        "timezone": "Asia/Kolkata",
        "display_name": "Sassoon Dock Fishing Harbour, Mumbai, Maharashtra, India",
    },
    {
        "id": 9008,
        "name": "Thoppumpady Fishing Harbour",
        "region": "Kochi, Kerala",
        "country": "India",
        "latitude": 9.9400,
        "longitude": 76.2600,
        "timezone": "Asia/Kolkata",
        "display_name": "Thoppumpady Fishing Harbour, Kochi, Kerala, India",
    },
    {
        "id": 9009,
        "name": "Visakhapatnam Fishing Harbour",
        "region": "Visakhapatnam, Andhra Pradesh",
        "country": "India",
        "latitude": 17.6900,
        "longitude": 83.3000,
        "timezone": "Asia/Kolkata",
        "display_name": "Visakhapatnam Fishing Harbour, Andhra Pradesh, India",
    },
]


class LocationService:
    """
    Service providing debounced geocoding search using Open-Meteo Geocoding API
    supplemented by exact maritime harbour definitions.
    """

    async def search_locations(self, query: str, count: int = 10) -> Dict[str, Any]:
        cleaned_query = query.strip()
        if not cleaned_query or len(cleaned_query) < 2:
            return {"query": query, "total": 0, "results": []}

        matched_harbours: List[Dict[str, Any]] = []
        q_lower = cleaned_query.lower()

        # 1. Match curated maritime harbour overlay first
        for h in MARITIME_HARBOUR_OVERLAY:
            if q_lower in h["name"].lower() or q_lower in h["region"].lower():
                matched_harbours.append(h)

        # 2. Query Open-Meteo Geocoding API
        api_results: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                params = {
                    "name": cleaned_query,
                    "count": count,
                    "language": "en",
                    "format": "json",
                }
                res = await client.get(OPEN_METEO_GEOCODING_URL, params=params)
                if res.status_code == 200:
                    raw_results = res.json().get("results", []) or []
                    for item in raw_results:
                        region = item.get("admin1") or item.get("admin2") or ""
                        country = item.get("country") or ""
                        display_parts = [item.get("name")]
                        if region:
                            display_parts.append(region)
                        if country:
                            display_parts.append(country)

                        api_results.append({
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "region": region,
                            "country": country,
                            "latitude": float(item.get("latitude")),
                            "longitude": float(item.get("longitude")),
                            "timezone": item.get("timezone", "auto"),
                            "display_name": ", ".join(display_parts),
                        })
        except Exception as e:
            logger.warning(f"Open-Meteo geocoding request failed for query '{cleaned_query}': {e}")

        # Combine harbour overlay matches (prioritized) + Open-Meteo API results (deduplicated)
        combined: List[Dict[str, Any]] = list(matched_harbours)
        seen_coords = {(round(h["latitude"], 3), round(h["longitude"], 3)) for h in matched_harbours}

        for item in api_results:
            coord_key = (round(item["latitude"], 3), round(item["longitude"], 3))
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                combined.append(item)

        final_list = combined[:count]
        return {
            "query": cleaned_query,
            "total": len(final_list),
            "results": final_list,
        }
