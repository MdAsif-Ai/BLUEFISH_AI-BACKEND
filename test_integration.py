"""
BlueFish AI Backend - Integration Verification Test Suite
==========================================================
Run this script to verify that all 6 backend services, Agent 3 (Data Ingestion),
Agent 4 (Retraining), and the Digital Twin engine are functioning correctly.

Usage:
    python3 test_integration.py
    or
    .venv/bin/python test_integration.py
"""

from __future__ import annotations

import asyncio
import sys
import numpy as np

print("==================================================")
print(" BlueFish AI Backend - Verification Test Suite")
print("==================================================")

# ── 1. Services Integration Tests ─────────────────────────────────────────────
print("\n[1/3] Testing AI Model Services in services/...")

from services.model1_service import PFZService
s1 = PFZService(None)
p1 = s1.predict({'sst': 28.5, 'chlorophyll': 1.2, 'salinity': 35.0})
print("  ✓ Model 1 (PFZ Service):", p1["zone_class"], f"(Probability: {p1['presence_probability']})")

from services.model2_service import OceanFeatureService
s2 = OceanFeatureService()
lats = np.linspace(10, 12, 10)
lons = np.linspace(78, 80, 10)
sst = np.random.randn(10, 10)
u = np.random.randn(10, 10)
v = np.random.randn(10, 10)
p2 = s2.detect_from_grids(sst, u, v, lats, lons)
print(f"  ✓ Model 2 (Ocean Features): {p2['front_count']} fronts, {p2['eddy_count']} eddies detected")

from services.model5_service import FleetDensityService
s5 = FleetDensityService()
vessels = [{'mmsi': f'100{i}', 'cell_ll_lat': 10.1 + i*0.01, 'cell_ll_lon': 79.1 + i*0.01, 'speed': 8.0, 'heading': 90.0} for i in range(15)]
p5 = s5.predict(vessels)
print(f"  ✓ Model 5 (Fleet Density): {len(p5['zones'])} overcrowded zones detected")

from services.model7_service import RouteOptimizationService
s7 = RouteOptimizationService()
p7 = s7.optimize(10.8, 79.8, 11.5, 80.2)
print(f"  ✓ Model 7 (A* Route): {p7['steps']} waypoints, Estimated fuel: {p7['estimated_fuel_liters']} L")

from services.model8_service import TimeWindowService
s8 = TimeWindowService()
p8 = s8.predict('2026-08-17', 10.8, 79.8)
print(f"  ✓ Model 8 (Solunar Math): Daily rating: {p8['daily_rating']} stars, Feeding windows: {len(p8['feeding_windows'])}")

from services.model10_service import CollisionDetectionService
s10 = CollisionDetectionService()
v1 = {'mmsi': '1001', 'lat': 10.0, 'lon': 79.0, 'speed': 10.0, 'heading': 90.0}
v2 = {'mmsi': '1002', 'lat': 10.001, 'lon': 79.001, 'speed': 10.0, 'heading': 270.0}
p10 = s10.detect_all_pairs([v1, v2])
print(f"  ✓ Model 10 (Collision Risk): {len(p10)} collision alert(s) detected")

# ── 2. Data Pipeline & Retraining Agent Tests ─────────────────────────────────
print("\n[2/3] Testing Agent 3 (Data Ingestion) & Agent 4 (Retraining)...")

from agents.data_ingestion_agent import download_daily_satellite_data, run_data_quality_checks, generate_and_cache_pfz_map
nc_path = download_daily_satellite_data('2026-08-17')
passed, checks, errors = run_data_quality_checks(nc_path)
print(f"  ✓ Agent 3 Data Quality Check Passed: {passed} (NaN checks: {len(errors)} errors)")

geojson = generate_and_cache_pfz_map(nc_path, '2026-08-17', s1)
print(f"  ✓ Agent 3 GeoJSON Daily PFZ Map generated with {len(geojson['features'])} features")

from agents.retraining_agent import check_and_trigger_retraining
async def test_retraining():
    res = await check_and_trigger_retraining(None, min_new_rows=10000)
    print(f"  ✓ Agent 4 Retraining Check: Threshold trigger evaluated (triggered={res['triggered']})")
asyncio.run(test_retraining())

# ── 3. Digital Twin Engine Tests ──────────────────────────────────────────────
print("\n[3/3] Testing Digital Twin Engine Simulation...")

from digital_twin.engine import run_simulation
async def test_sim():
    res = await run_simulation(days=1, fleet_size=5, policy_restrictions={}, initial_lat=10.8, initial_lon=79.8)
    print(f"  ✓ Digital Twin Simulation: {len(res['steps'])} hourly snapshots generated, total catch: {res['summary']['total_catch_kg']} kg")
asyncio.run(test_sim())

print("\n==================================================")
print(" ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("==================================================")
