from services.model9_service import VesselSegmentationService

# Instantiate Model 9 service (with None for fallback mode)
service = VesselSegmentationService(kmeans_model=None)

# Sample vessel profile
vessel_data = {
    "mmsi": "367999888",
    "avg_trip_duration_hours": 85.0,
    "max_distance_from_port_km": 180.0,
    "avg_fuel_per_trip_liters": 750.0,
    "avg_catch_per_trip_kg": 1200.0,
    "night_fishing_ratio": 0.4,
}

# Single prediction
pred = service.predict(vessel_data)
print("Single Prediction:")
print(pred)

# Fleet composition test
fleet = [
    {"mmsi": "101", "avg_trip_duration_hours": 10.0, "max_distance_from_port_km": 15.0},
    {"mmsi": "102", "avg_trip_duration_hours": 90.0, "max_distance_from_port_km": 200.0},
    {"mmsi": "103", "night_fishing_ratio": 0.7},
]

print("\nFleet Composition Breakdown:")
print(service.get_fleet_composition(fleet))
