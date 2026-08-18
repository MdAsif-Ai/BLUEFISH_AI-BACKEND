import asyncio
from services.live_data_service import LiveDataService
from core.model_loader import get_model_registry

async def main():
    print("Testing 97 Live Telemetry Feed across all 11 Models...")
    svc = LiveDataService()
    live_res = await svc.get_live_data(13.1167, 80.2833, "Kasimedu Harbour")
    
    features_97 = live_res.get("master_features_97", {})
    print(f"Retrieved {len(features_97)} live features from LiveDataService.")
    
    reg = get_model_registry()
    print("Model registry initialized. Testing individual service invocations...")
    
    # 1. Model 1 - PFZ
    try:
        from services.model1_service import PFZPredictorService
        m1_svc = PFZPredictorService(reg.model1_rf, reg.model1_scaler)
        m1_res = m1_svc.predict_from_features(features_97)
        print("Model 1 Output:", m1_res)
    except Exception as e:
        print("Model 1 Error:", e)

    # 2. Model 2 - Catch Biomass
    try:
        from services.model2_service import CatchPredictorService
        m2_svc = CatchPredictorService(reg.model2_rf)
        m2_res = m2_svc.predict_from_features(features_97)
        print("Model 2 Output:", m2_res)
    except Exception as e:
        print("Model 2 Error:", e)

    # 3. Model 3 - Migration LSTM
    try:
        from services.model3_service import MigrationPredictorService
        m3_svc = MigrationPredictorService(reg.model3)
        m3_res = m3_svc.predict_from_features(features_97)
        print("Model 3 Output:", m3_res)
    except Exception as e:
        print("Model 3 Error:", e)

    # 4. Model 4 - Seasonal TFT
    try:
        from services.model4_service import SeasonalForecastService
        m4_svc = SeasonalForecastService(reg.model4)
        m4_res = m4_svc.predict_from_features(features_97)
        print("Model 4 Output:", m4_res)
    except Exception as e:
        print("Model 4 Error:", e)

    # 5. Model 5 - Route Optimization
    try:
        from services.model5_service import RouteOptimizationService
        m5_svc = RouteOptimizationService()
        m5_res = m5_svc.predict_from_features(features_97)
        print("Model 5 Output:", m5_res)
    except Exception as e:
        print("Model 5 Error:", e)

    # 6. Model 6 - Dark Fleet Anomaly
    try:
        from services.model6_service import AnomalyDetectorService
        m6_svc = AnomalyDetectorService(reg.model6_dbscan, reg.model6_isolation)
        m6_res = m6_svc.predict_from_features(features_97)
        print("Model 6 Output:", m6_res)
    except Exception as e:
        print("Model 6 Error:", e)

    # 7. Model 7 - Fleet Profiler K-Means
    try:
        from services.model7_service import FleetProfilerService
        m7_svc = FleetProfilerService(reg.model7_kmeans, reg.model7_scaler)
        m7_res = m7_svc.predict_from_features(features_97)
        print("Model 7 Output:", m7_res)
    except Exception as e:
        print("Model 7 Error:", e)

    # 8. Model 8 - Digital Twin PDE Simulator
    try:
        from services.model8_service import DigitalTwinService
        m8_svc = DigitalTwinService()
        m8_res = m8_svc.predict_from_features(features_97)
        print("Model 8 Output:", m8_res)
    except Exception as e:
        print("Model 8 Error:", e)

    # 9. Model 9 - Fleet Command LLM
    try:
        from services.model9_service import CommandAgentService
        m9_svc = CommandAgentService()
        m9_res = m9_svc.predict_from_features(features_97)
        print("Model 9 Output:", m9_res)
    except Exception as e:
        print("Model 9 Error:", e)

    # 10. Model 10 - MLOps Drift Monitor
    try:
        from services.model10_service import MLOpsService
        m10_svc = MLOpsService()
        m10_res = m10_svc.predict_from_features(features_97)
        print("Model 10 Output:", m10_res)
    except Exception as e:
        print("Model 10 Error:", e)

    # 11. Model 11 - Climate Risk XGBoost
    try:
        from services.model11_service import ClimateRiskService
        m11_svc = ClimateRiskService(reg.model11_xgb, reg.model11_scaler)
        m11_res = m11_svc.predict_from_features(features_97)
        print("Model 11 Output:", m11_res)
    except Exception as e:
        print("Model 11 Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
