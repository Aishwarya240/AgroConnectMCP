import sys
from server import get_weather_forecast

def test_weather_prediction():
    print("Testing agricultural weather forecast tool...")
    # Target Rome coordinates (Lat 41.9, Lon 12.5) for testing
    try:
        report = get_weather_forecast(latitude=41.9, longitude=12.5, forecast_days=3)
        print("\n--- WEATHER FORECAST REPORT ---")
        print(report)
        print("-------------------------------\n")
        if "Agricultural Weather Report" in report and "Error" not in report:
            print("SUCCESS: Weather forecasting tool executed successfully!")
            return True
        else:
            print(f"FAILURE: Report has unexpected structure or contains error: {report}")
            return False
    except Exception as e:
        print(f"FAILURE: Weather tool test raised exception: {e}")
        return False

if __name__ == "__main__":
    success = test_weather_prediction()
    if not success:
        sys.exit(1)
