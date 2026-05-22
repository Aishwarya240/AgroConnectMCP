import sys
from server import get_soil_conditions, get_historical_weather

def test_soil_and_history():
    print("Testing Soil Conditions and Historical Weather tools...")
    
    # Coordinates for Rome (Lat 41.9, Lon 12.5)
    lat, lon = 41.9, 12.5
    
    # 1. Test Soil Conditions
    try:
        print("\n--- TESTING SOIL CONDITIONS TOOL ---")
        soil_report = get_soil_conditions(latitude=lat, longitude=lon, days=2)
        print(soil_report)
        print("------------------------------------\n")
        
        if "Soil Condition Report" not in soil_report or "Error" in soil_report:
            print("FAILURE: Soil Conditions tool returned an error or invalid structure.")
            return False
            
    except Exception as e:
        print(f"FAILURE: Soil Conditions tool test raised exception: {e}")
        return False
        
    # 2. Test Historical Weather
    try:
        print("\n--- TESTING HISTORICAL WEATHER TOOL ---")
        # May 1st to May 5th, 2025 (Well in the past, highly cached or archives ready)
        history_report = get_historical_weather(latitude=lat, longitude=lon, start_date="2025-05-01", end_date="2025-05-05")
        print(history_report)
        print("---------------------------------------\n")
        
        if "Historical Weather Assessment Report" not in history_report or "Error" in history_report:
            print("FAILURE: Historical Weather tool returned an error or invalid structure.")
            return False
            
    except Exception as e:
        print(f"FAILURE: Historical Weather tool test raised exception: {e}")
        return False

    print("SUCCESS: Both Soil Conditions and Historical Weather tools executed successfully!")
    return True

if __name__ == "__main__":
    success = test_soil_and_history()
    if not success:
        sys.exit(1)
