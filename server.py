import os
import base64
import mimetypes
import logging
import json
import datetime
import math
from typing import Optional
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgroConnectMCP")

# Load environment variables from .env file
load_dotenv()

# Setup Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Initialize FastMCP Server
mcp = FastMCP("AgroConnectMCP")


# --- JSON ENCODER ---

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
        return super(DateTimeEncoder, self).default(obj)


# --- HELPER: Open-Meteo hourly query ---

def _query_hourly(latitude: float, longitude: float, variables: list, forecast_days: int = 7) -> dict:
    """Internal helper to query Open-Meteo hourly endpoint and return location + DataFrame."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": variables,
        "timezone": "auto",
        "forecast_days": forecast_days
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    for idx, var_name in enumerate(variables):
        hourly_data[var_name] = hourly.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    location = {
        "latitude": float(response.Latitude()),
        "longitude": float(response.Longitude()),
        "elevation": float(response.Elevation()),
        "utc_offset_seconds": int(response.UtcOffsetSeconds())
    }
    return {"location": location, "df": df}


def _query_daily(latitude: float, longitude: float, variables: list, forecast_days: int = 7, int64_vars: list = None) -> dict:
    """Internal helper to query Open-Meteo daily endpoint and return location + DataFrame."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": variables,
        "timezone": "auto",
        "forecast_days": forecast_days
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }
    int64_vars = int64_vars or []
    for idx, var_name in enumerate(variables):
        if var_name in int64_vars:
            daily_data[var_name] = pd.to_datetime(daily.Variables(idx).ValuesInt64AsNumpy(), unit="s", utc=True)
        else:
            daily_data[var_name] = daily.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data=daily_data)
    location = {
        "latitude": float(response.Latitude()),
        "longitude": float(response.Longitude()),
        "elevation": float(response.Elevation()),
        "utc_offset_seconds": int(response.UtcOffsetSeconds())
    }
    return {"location": location, "df": df}


def _format_result(location: dict, key: str, df: pd.DataFrame) -> str:
    """Format location + DataFrame into a JSON string."""
    result = {
        "location": location,
        key: json.loads(df.to_json(orient="records", date_format="iso"))
    }
    return json.dumps(result, cls=DateTimeEncoder)


# --- HELPER: Image processing ---

def get_base64_image(image_data: str) -> str:
    """
    Helper function to normalize and convert the input image_data to a base64 Data URI.
    Supports:
    - Base64 data URI (starts with 'data:image/')
    - Raw Base64 string (no prefix)
    - Local file path
    - Remote URL (starts with 'http://' or 'https://')
    """
    if image_data.strip().startswith("data:image/"):
        return image_data.strip()

    if image_data.strip().startswith(("http://", "https://")):
        logger.info(f"Downloading image from URL: {image_data}")
        try:
            response = requests.get(image_data.strip(), timeout=10)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "image/jpeg")
            encoded_str = base64.b64encode(response.content).decode("utf-8")
            return f"data:{content_type};base64,{encoded_str}"
        except Exception as e:
            raise ValueError(f"Failed to fetch image from URL '{image_data}': {str(e)}")

    if os.path.exists(image_data):
        logger.info(f"Reading image from local path: {image_data}")
        try:
            mime_type, _ = mimetypes.guess_type(image_data)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"
            with open(image_data, "rb") as image_file:
                encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_str}"
        except Exception as e:
            raise ValueError(f"Failed to read local image file '{image_data}': {str(e)}")

    try:
        cleaned = image_data.strip()
        padding_needed = len(cleaned) % 4
        if padding_needed:
            cleaned += "=" * (4 - padding_needed)
        base64.b64decode(cleaned)
        return f"data:image/jpeg;base64,{cleaned}"
    except Exception:
        raise ValueError(
            "Invalid image input. The input must be a valid file path, "
            "an http/https image URL, a base64 Data URI, or a raw base64 string."
        )


# ============================================================
# TOOL 1: Plant Disease Prediction
# ============================================================

@mcp.tool
def predict_plant_disease(
    image_data: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> str:
    """
    Predict and identify plant/crop diseases from an image using the Kindwise crop.health API.

    Args:
        image_data (str): The plant/leaf image to analyze. Can be:
            - A base64-encoded Data URI (starts with 'data:image/')
            - A raw base64 string
            - A local file path on the system
            - An HTTP/HTTPS URL pointing to an image
        latitude (float, optional): Latitude coordinates of the crop's location for better geo-diagnosis.
        longitude (float, optional): Longitude coordinates of the crop's location for better geo-diagnosis.
    """
    api_key = os.getenv("KINDWISE_API_KEY")
    if not api_key:
        return (
            "Error: KINDWISE_API_KEY environment variable is not configured. "
            "Please check that your .env file is present and properly set up."
        )

    try:
        formatted_image = get_base64_image(image_data)
    except ValueError as e:
        return f"Error processing image input: {str(e)}"

    url = "https://crop.kindwise.com/api/v1/identification"
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "images": [formatted_image],
        "similar_images": True
    }
    if latitude is not None:
        payload["latitude"] = latitude
    if longitude is not None:
        payload["longitude"] = longitude

    logger.info("Sending request to Kindwise crop.health API...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 401:
            return "Error: Unauthorized. The configured KINDWISE_API_KEY is invalid or inactive."
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed: {str(e)}")
        try:
            err_json = response.json()
            err_msg = err_json.get("message") or err_json.get("error", {}).get("message")
            if err_msg:
                return f"API Error ({response.status_code}): {err_msg}"
        except Exception:
            pass
        return f"Failed to connect to the Crop Disease Prediction API: {str(e)}"

    try:
        data = response.json()
    except Exception as e:
        return f"Failed to parse response from Crop API: {str(e)}"

    output_lines = ["# AgroConnectMCP Plant Disease Assessment Report\n"]
    is_plant = data.get("is_plant")
    if is_plant is False:
        output_lines.append("> [!WARNING]\n> The Crop API did not detect a plant in this image. Results may not be accurate.\n")

    result = data.get("result", {})
    disease_info = result.get("disease", {})
    suggestions = disease_info.get("suggestions", [])

    if not suggestions:
        output_lines.append("No specific plant diseases or pests were identified. The plant might be healthy, or the image quality was insufficient for diagnosis.")
        return "\n".join(output_lines)

    output_lines.append("## Identified Condition Suggestions\n")
    for idx, sug in enumerate(suggestions, 1):
        name = sug.get("name", "Unknown Condition")
        prob = sug.get("probability", 0.0)
        details = sug.get("details", {})
        common_names = details.get("common_names", [])
        description = details.get("description", "No description available.")
        treatment = details.get("treatment", {})
        severity = details.get("severity", "Unknown")
        symptoms = details.get("symptoms", [])

        output_lines.append(f"### {idx}. {name}")
        output_lines.append(f"- **Confidence/Probability**: {prob * 100:.1f}%")
        output_lines.append(f"- **Severity Level**: {severity.capitalize()}")
        if common_names:
            output_lines.append(f"- **Common Names**: {', '.join(common_names)}")
        output_lines.append(f"\n#### Description\n{description}\n")
        if symptoms:
            output_lines.append("#### Key Symptoms")
            for sym in symptoms:
                output_lines.append(f"- {sym}")
            output_lines.append("")
        if treatment:
            output_lines.append("#### Recommended Treatment & Prevention Options")
            if isinstance(treatment, dict):
                for category, recommendations in treatment.items():
                    if recommendations:
                        cat_formatted = category.replace("_", " ").capitalize()
                        output_lines.append(f"##### {cat_formatted}")
                        if isinstance(recommendations, list):
                            for rec in recommendations:
                                output_lines.append(f"- {rec}")
                        else:
                            output_lines.append(f"{recommendations}")
            elif isinstance(treatment, list):
                for rec in treatment:
                    output_lines.append(f"- {rec}")
            else:
                output_lines.append(f"{treatment}")
            output_lines.append("")
        output_lines.append("---\n")

    return "\n".join(output_lines)


# ============================================================
# TOOL 2: Current Weather
# ============================================================

@mcp.tool
def get_current_weather(
    latitude: float,
    longitude: float
) -> str:
    """
    Get current weather conditions including temperature, humidity, rain, pressure, wind speed and direction.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "rain", "is_day", "pressure_msl", "wind_speed_10m", "wind_direction_10m"],
        "timezone": "auto"
    }

    logger.info(f"Querying current weather for ({latitude}, {longitude})...")
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    current = response.Current()
    result = {
        "location": {
            "latitude": float(response.Latitude()),
            "longitude": float(response.Longitude()),
            "elevation": float(response.Elevation()),
            "utc_offset_seconds": int(response.UtcOffsetSeconds())
        },
        "current": {
            "time": pd.to_datetime(current.Time(), unit="s", utc=True).isoformat(),
            "temperature_2m": float(current.Variables(0).Value()),
            "relative_humidity_2m": float(current.Variables(1).Value()),
            "rain": float(current.Variables(2).Value()),
            "is_day": float(current.Variables(3).Value()),
            "pressure_msl": float(current.Variables(4).Value()),
            "wind_speed_10m": float(current.Variables(5).Value()),
            "wind_direction_10m": float(current.Variables(6).Value())
        }
    }
    return json.dumps(result, cls=DateTimeEncoder)


# ============================================================
# TOOL 3: Temperature Forecast (hourly at 2m, 80m, 120m, 180m + apparent)
# ============================================================

@mcp.tool
def get_temperature_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly temperature forecast at multiple heights (2m, 80m, 120m, 180m) and apparent (feels-like) temperature.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["temperature_2m", "apparent_temperature", "temperature_80m", "temperature_120m", "temperature_180m"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "temperature_hourly", data["df"])


# ============================================================
# TOOL 4: Humidity Forecast (relative humidity, dew point, vapour pressure deficit)
# ============================================================

@mcp.tool
def get_humidity_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly humidity forecast including relative humidity, dew point, and vapour pressure deficit.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "humidity_hourly", data["df"])


# ============================================================
# TOOL 5: Precipitation Forecast (probability, amount, rain, showers, snowfall, snow depth)
# ============================================================

@mcp.tool
def get_precipitation_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly precipitation forecast including probability, total precipitation, rain, showers, snowfall, and snow depth.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "precipitation_hourly", data["df"])


# ============================================================
# TOOL 6: Wind Forecast (speed at 10m/80m/120m/180m, direction at all heights, gusts)
# ============================================================

@mcp.tool
def get_wind_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly wind forecast including speed at 10m/80m/120m/180m, direction at all heights, and wind gusts.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = [
        "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
        "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
        "wind_gusts_10m"
    ]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "wind_hourly", data["df"])


# ============================================================
# TOOL 7: Pressure Forecast (mean sea level pressure, surface pressure)
# ============================================================

@mcp.tool
def get_pressure_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly atmospheric pressure forecast including mean sea level pressure and surface pressure.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["pressure_msl", "surface_pressure"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "pressure_hourly", data["df"])


# ============================================================
# TOOL 8: Cloud Cover Forecast (total, low, mid, high)
# ============================================================

@mcp.tool
def get_cloud_cover_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly cloud cover forecast at all altitude layers (total, low, mid, high).

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "cloud_cover_hourly", data["df"])


# ============================================================
# TOOL 9: Visibility Forecast
# ============================================================

@mcp.tool
def get_visibility_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly visibility forecast in meters.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["visibility"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "visibility_hourly", data["df"])


# ============================================================
# TOOL 10: Evapotranspiration Forecast (ET and ET0 FAO reference)
# ============================================================

@mcp.tool
def get_evapotranspiration_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly evapotranspiration forecast including actual evapotranspiration and ET0 FAO reference evapotranspiration.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["evapotranspiration", "et0_fao_evapotranspiration"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "evapotranspiration_hourly", data["df"])


# ============================================================
# TOOL 11: Weather Code Forecast
# ============================================================

@mcp.tool
def get_weather_code_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly WMO weather interpretation codes. Codes indicate conditions like clear sky (0), fog (45/48), drizzle (51-57), rain (61-67), snow (71-77), thunderstorm (95-99), etc.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["weather_code"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "weather_code_hourly", data["df"])


# ============================================================
# TOOL 12: UV Index Forecast (daily max)
# ============================================================

@mcp.tool
def get_uv_index_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get daily maximum UV index forecast.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["uv_index_max"]
    try:
        data = _query_daily(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "uv_index_daily", data["df"])


# ============================================================
# TOOL 13: Sunrise & Sunset Times (daily)
# ============================================================

@mcp.tool
def get_sunrise_sunset(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get daily sunrise and sunset times.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["sunrise", "sunset"]
    try:
        data = _query_daily(latitude, longitude, variables, forecast_days, int64_vars=["sunrise", "sunset"])
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "sunrise_sunset_daily", data["df"])


# ============================================================
# TOOL 14: Daily Temperature Forecast (max & min)
# ============================================================

@mcp.tool
def get_daily_temperature(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get daily maximum and minimum temperature forecast.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["temperature_2m_max", "temperature_2m_min"]
    try:
        data = _query_daily(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "temperature_daily", data["df"])


# ============================================================
# TOOL 15: Daily Rain Forecast (rain sum)
# ============================================================

@mcp.tool
def get_daily_rain(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get daily total rainfall sum forecast.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["rain_sum"]
    try:
        data = _query_daily(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "rain_daily", data["df"])


# ============================================================
# TOOL 16: Daily Wind Forecast (max speed & max gusts)
# ============================================================

@mcp.tool
def get_daily_wind(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get daily maximum wind speed and maximum wind gusts forecast.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["wind_speed_10m_max", "wind_gusts_10m_max"]
    try:
        data = _query_daily(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "wind_daily", data["df"])


# ============================================================
# TOOL 17: Soil Temperature Forecast (0cm, 6cm, 18cm, 54cm)
# ============================================================

@mcp.tool
def get_soil_temperature(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly soil temperature forecast at multiple depths (0cm, 6cm, 18cm, 54cm).

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "soil_temperature_hourly", data["df"])


# ============================================================
# TOOL 18: Soil Moisture Forecast (0-1cm, 1-3cm, 3-9cm, 9-27cm)
# ============================================================

@mcp.tool
def get_soil_moisture(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get hourly soil moisture forecast at multiple depths (0-1cm, 1-3cm, 3-9cm, 9-27cm).

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return json.dumps({"error": "forecast_days must be between 1 and 14."})

    variables = ["soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm"]
    try:
        data = _query_hourly(latitude, longitude, variables, forecast_days)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})

    return _format_result(data["location"], "soil_moisture_hourly", data["df"])


# ============================================================
# TOOL 19: Historical Weather
# ============================================================

@mcp.tool
def get_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str
) -> str:
    """
    Retrieve historical weather data for a past timeframe including daily max/min temperature, precipitation, and wind speed.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        start_date (str): Start date of the range (YYYY-MM-DD).
        end_date (str): End date of the range (YYYY-MM-DD).
    """
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"error": "Invalid date format. Please use YYYY-MM-DD (e.g. '2025-05-01')."})

    if start_dt > end_dt:
        return json.dumps({"error": "start_date must be before or equal to end_date."})

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "auto"
    }

    logger.info(f"Querying historical weather for ({latitude}, {longitude}) from {start_date} to {end_date}...")
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch historical data: {str(e)}"})

    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }
    for idx, var_name in enumerate(params["daily"]):
        daily_data[var_name] = daily.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data=daily_data)
    result = {
        "location": {
            "latitude": float(response.Latitude()),
            "longitude": float(response.Longitude()),
            "elevation": float(response.Elevation()),
            "utc_offset_seconds": int(response.UtcOffsetSeconds())
        },
        "historical_daily": json.loads(df.to_json(orient="records", date_format="iso"))
    }
    return json.dumps(result, cls=DateTimeEncoder)


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger.info("Starting AgroConnectMCP Server...")
    mcp.run()
