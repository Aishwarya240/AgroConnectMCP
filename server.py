import os
import base64
import mimetypes
import logging
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


def get_base64_image(image_data: str) -> str:
    """
    Helper function to normalize and convert the input image_data to a base64 Data URI.
    Supports:
    - Base64 data URI (starts with 'data:image/')
    - Raw Base64 string (no prefix)
    - Local file path
    - Remote URL (starts with 'http://' or 'https://')
    """
    # 1. Check if it's already a Data URI
    if image_data.strip().startswith("data:image/"):
        return image_data.strip()

    # 2. Check if it's a remote URL
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

    # 3. Check if it's a local file path
    if os.path.exists(image_data):
        logger.info(f"Reading image from local path: {image_data}")
        try:
            mime_type, _ = mimetypes.guess_type(image_data)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"  # Default fallback
            
            with open(image_data, "rb") as image_file:
                encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_str}"
        except Exception as e:
            raise ValueError(f"Failed to read local image file '{image_data}': {str(e)}")

    # 4. Assume it's a raw base64 string
    # Validate raw base64 encoding to make sure it's valid
    try:
        # Strip potential whitespace
        cleaned = image_data.strip()
        # Add padding if missing (often a case with raw base64 strings)
        padding_needed = len(cleaned) % 4
        if padding_needed:
            cleaned += "=" * (4 - padding_needed)
        base64.b64decode(cleaned)
        # Standard fallback is jpeg mime-type
        return f"data:image/jpeg;base64,{cleaned}"
    except Exception:
        raise ValueError(
            "Invalid image input. The input must be a valid file path, "
            "an http/https image URL, a base64 Data URI, or a raw base64 string."
        )

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
    # 1. Fetch API Key
    api_key = os.getenv("KINDWISE_API_KEY")
    if not api_key:
        return (
            "Error: KINDWISE_API_KEY environment variable is not configured. "
            "Please check that your .env file is present and properly set up."
        )

    # 2. Get normalized base64 image data
    try:
        formatted_image = get_base64_image(image_data)
    except ValueError as e:
        return f"Error processing image input: {str(e)}"

    # 3. Build request payload
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

    # 4. Make Request
    logger.info("Sending request to Kindwise crop.health API...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        # Handle HTTP errors
        if response.status_code == 401:
            return "Error: Unauthorized. The configured KINDWISE_API_KEY is invalid or inactive."
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed: {str(e)}")
        # Check if we have error details from the server
        try:
            err_json = response.json()
            err_msg = err_json.get("message") or err_json.get("error", {}).get("message")
            if err_msg:
                return f"API Error ({response.status_code}): {err_msg}"
        except Exception:
            pass
        return f"Failed to connect to the Crop Disease Prediction API: {str(e)}"

    # 5. Parse Response
    try:
        data = response.json()
    except Exception as e:
        return f"Failed to parse response from Crop API: {str(e)}"

    # 6. Format Output
    output_lines = ["# AgroConnectMCP Plant Disease Assessment Report\n"]
    
    # Check general properties
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
        
        # Symptoms formatting
        if symptoms:
            output_lines.append("#### Key Symptoms")
            for sym in symptoms:
                output_lines.append(f"- {sym}")
            output_lines.append("")
            
        # Treatment recommendations
        if treatment:
            output_lines.append("#### Recommended Treatment & Prevention Options")
            # treatment can be a string, a list, or a dict containing biological, chemical, prevention etc.
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

@mcp.tool
def get_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Retrieve an agricultural weather forecast and assessment report for given coordinates.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        forecast_days (int): Number of days to forecast (1 to 14, default is 7).
    """
    if forecast_days < 1 or forecast_days > 14:
        return "Error: forecast_days must be between 1 and 14."

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "soil_temperature_0_to_10cm",
            "soil_moisture_0_to_10cm",
            "evapotranspiration"
        ],
        "timezone": "auto",
        "forecast_days": forecast_days
    }

    logger.info(f"Querying weather forecast for coordinates: ({latitude}, {longitude}) for {forecast_days} days...")
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        logger.error(f"Open-Meteo API query failed: {str(e)}")
        return f"Error: Failed to fetch weather data from Open-Meteo API: {str(e)}"

    # Process hourly data
    hourly = response.Hourly()
    utc_offset = response.UtcOffsetSeconds()
    
    try:
        # Load hourly variables as numpy arrays
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "precipitation": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
            "wind_dir": hourly.Variables(4).ValuesAsNumpy(),
            "soil_temp": hourly.Variables(5).ValuesAsNumpy(),
            "soil_moisture": hourly.Variables(6).ValuesAsNumpy(),
            "evapotranspiration": hourly.Variables(7).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(data=hourly_data)
        # Shift to local timezone
        df["date_local"] = df["date"] + pd.to_timedelta(utc_offset, unit="s")
    except Exception as e:
        logger.error(f"Error compiling pandas data: {str(e)}")
        return f"Error: Failed to process weather data: {str(e)}"

    # Perform Agricultural Assessments
    # Calculate general risk indices
    high_humidity_streak = 0
    max_humidity_streak = 0
    for h in df["humidity"]:
        if h > 85:
            high_humidity_streak += 1
            max_humidity_streak = max(max_humidity_streak, high_humidity_streak)
        else:
            high_humidity_streak = 0
            
    # Spraying suitability ratio (hours with 5 <= wind <= 15)
    total_hours = len(df)
    optimal_spray_hours = len(df[(df["wind_speed"] >= 5) & (df["wind_speed"] <= 15)])
    spray_ratio = (optimal_spray_hours / total_hours) * 100
    
    # Soil conditions
    avg_soil_temp = df["soil_temp"].mean()
    avg_soil_moist = df["soil_moisture"].mean()
    
    # We will resample/group by day based on local date
    df["day"] = df["date_local"].dt.date
    
    # Daily aggregation
    daily_agg = df.groupby("day").agg(
        temp_max=("temp", "max"),
        temp_min=("temp", "min"),
        humidity_avg=("humidity", "mean"),
        precip_sum=("precipitation", "sum"),
        wind_max=("wind_speed", "max"),
        soil_temp_avg=("soil_temp", "mean"),
        soil_moist_avg=("soil_moisture", "mean"),
        et_sum=("evapotranspiration", "sum")
    ).reset_index()
    
    output = []
    output.append(f"# AgroConnectMCP Agricultural Weather Report\n")
    output.append(f"- **Coordinates**: {response.Latitude():.4f}°N, {response.Longitude():.4f}°E")
    output.append(f"- **Elevation**: {response.Elevation()} m above sea level")
    output.append(f"- **Timezone Offset**: {utc_offset / 3600:+.1f} hours from UTC")
    output.append(f"- **Forecast Period**: {forecast_days} days\n")
    
    # Risk summary card
    output.append("## 🌾 Agricultural Advisory Summary\n")
    
    # Disease risk advisory
    if max_humidity_streak >= 8:
        output.append("> [!WARNING]")
        output.append(f"> **High Fungal Disease Risk**: Sustained relative humidity (>85%) detected for up to {max_humidity_streak} consecutive hours. Favorable conditions for pathogens like Early Blight or powdery mildew. Consider proactive crop health treatments.")
    else:
        output.append("> [!NOTE]")
        output.append(f"> **Low Fungal Disease Risk**: Relative humidity levels are not sustained at extreme levels. Standard preventative measures are sufficient.")

    # Spraying advisory
    if spray_ratio > 60:
        output.append("> [!TIP]")
        output.append(f"> **Excellent Spraying Conditions**: {spray_ratio:.1f}% of forecast hours have optimal wind speeds (5-15 km/h). Safe for pesticide/herbicide application with minimal drift risk.")
    elif spray_ratio > 30:
        output.append("> [!IMPORTANT]")
        output.append(f"> **Moderate Spraying Conditions**: {spray_ratio:.1f}% of forecast hours are optimal. Check hourly forecast tables to target low-drift windows (avoid calm inversion periods and high winds).")
    else:
        output.append("> [!CAUTION]")
        output.append(f"> **Poor Spraying Conditions**: Only {spray_ratio:.1f}% of forecast hours are optimal. High wind speeds or prolonged calm spells (inversion risk) make spraying hazardous. Postpone chemical treatment if possible.")

    # Soil advisory
    if avg_soil_temp < 10.0:
        output.append("> [!WARNING]")
        output.append(f"> **Low Soil Temperature**: Average soil temperature is {avg_soil_temp:.1f}°C. Too cold for seed germination of warm-season crops (e.g., tomatoes, corn). Delay planting to avoid seed rot.")
    elif avg_soil_moist < 0.15:
        output.append("> [!IMPORTANT]")
        output.append(f"> **Dry Soil Warning**: Average soil moisture is low ({avg_soil_moist:.3f} m³/m³). Crops may experience drought stress. Irrigation is recommended.")
    elif avg_soil_moist > 0.45:
        output.append("> [!WARNING]")
        output.append(f"> **Waterlogged Soil**: Average soil moisture is very high ({avg_soil_moist:.3f} m³/m³). Risk of root rot and oxygen depletion in the root zone. Ensure adequate field drainage.")
    else:
        output.append("> [!TIP]")
        output.append(f"> **Optimal Soil Conditions**: Average soil temperature is {avg_soil_temp:.1f}°C and moisture is {avg_soil_moist:.3f} m³/m³. Highly favorable for root development and germination.")
        
    output.append("")

    # Daily Summary Table
    output.append("## 📅 Daily Agricultural Forecast Summary\n")
    output.append("| Date | Temp Max/Min (°C) | Avg RH (%) | Precip (mm) | Max Wind (km/h) | Avg Soil Temp (°C) | Avg Soil Moist (m³/m³) | Total ET (mm) |")
    output.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in daily_agg.iterrows():
        date_str = row['day'].strftime('%Y-%m-%d')
        output.append(
            f"| {date_str} "
            f"| {row['temp_max']:.1f}° / {row['temp_min']:.1f}° "
            f"| {row['humidity_avg']:.1f}% "
            f"| {row['precip_sum']:.1f} "
            f"| {row['wind_max']:.1f} "
            f"| {row['soil_temp_avg']:.1f}° "
            f"| {row['soil_moist_avg']:.3f} "
            f"| {row['et_sum']:.2f} |"
        )
    output.append("")

    # Hourly Details for first 24 hours
    output.append("## 🕐 24-Hour Detailed Agricultural Outlook\n")
    output.append("| Time | Temp (°C) | RH (%) | Wind (km/h) | Soil Temp (°C) | Soil Moist (m³/m³) | Spraying Suitability |")
    output.append("| --- | --- | --- | --- | --- | --- | --- |")
    
    # Slice the first 24 hours
    df_24h = df.head(24)
    for _, row in df_24h.iterrows():
        time_str = row['date_local'].strftime('%H:%M')
        
        # Spraying suitability tag
        w = row['wind_speed']
        if w < 5.0:
            suitability = "⚠️ Calm (Inversion)"
        elif w <= 15.0:
            suitability = "✅ Optimal"
        else:
            suitability = "❌ High Wind (Drift)"
            
        output.append(
            f"| {time_str} "
            f"| {row['temp']:.1f}° "
            f"| {row['humidity']:.0f}% "
            f"| {row['wind_speed']:.1f} "
            f"| {row['soil_temp']:.1f}° "
            f"| {row['soil_moisture']:.3f} "
            f"| {suitability} |"
        )
    
    output.append("")
    return "\n".join(output)

@mcp.tool
def get_soil_conditions(
    latitude: float,
    longitude: float,
    days: int = 3
) -> str:
    """
    Retrieve a comprehensive soil condition analysis at multiple depths (0-7cm, 7-28cm, 28-100cm, 100-255cm).

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        days (int): Number of days to analyze (1 to 14, default is 3).
    """
    if days < 1 or days > 14:
        return "Error: days must be between 1 and 14."

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm"
        ],
        "timezone": "auto",
        "forecast_days": days
    }

    logger.info(f"Querying soil conditions for coordinates: ({latitude}, {longitude}) for {days} days...")
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        logger.error(f"Open-Meteo API query for soil failed: {str(e)}")
        return f"Error: Failed to fetch soil data from Open-Meteo API: {str(e)}"

    hourly = response.Hourly()
    utc_offset = response.UtcOffsetSeconds()

    try:
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "soil_temp_1": hourly.Variables(0).ValuesAsNumpy(),
            "soil_temp_2": hourly.Variables(1).ValuesAsNumpy(),
            "soil_temp_3": hourly.Variables(2).ValuesAsNumpy(),
            "soil_temp_4": hourly.Variables(3).ValuesAsNumpy(),
            "soil_moist_1": hourly.Variables(4).ValuesAsNumpy(),
            "soil_moist_2": hourly.Variables(5).ValuesAsNumpy(),
            "soil_moist_3": hourly.Variables(6).ValuesAsNumpy(),
            "soil_moist_4": hourly.Variables(7).ValuesAsNumpy(),
        }
        df = pd.DataFrame(data=hourly_data)
        df["date_local"] = df["date"] + pd.to_timedelta(utc_offset, unit="s")
    except Exception as e:
        logger.error(f"Error compiling soil pandas data: {str(e)}")
        return f"Error: Failed to process soil data: {str(e)}"

    df["day"] = df["date_local"].dt.date
    daily_agg = df.groupby("day").agg(
        t1=("soil_temp_1", "mean"),
        t2=("soil_temp_2", "mean"),
        t3=("soil_temp_3", "mean"),
        t4=("soil_temp_4", "mean"),
        m1=("soil_moist_1", "mean"),
        m2=("soil_moist_2", "mean"),
        m3=("soil_moist_3", "mean"),
        m4=("soil_moist_4", "mean"),
    ).reset_index()

    # Calculate overall averages
    avg_t1 = df["soil_temp_1"].mean()
    avg_t2 = df["soil_temp_2"].mean()
    avg_t3 = df["soil_temp_3"].mean()
    avg_t4 = df["soil_temp_4"].mean()
    avg_m1 = df["soil_moist_1"].mean()
    avg_m2 = df["soil_moist_2"].mean()
    avg_m3 = df["soil_moist_3"].mean()
    avg_m4 = df["soil_moist_4"].mean()

    output = []
    output.append(f"# AgroConnectMCP Soil Condition Report\n")
    output.append(f"- **Coordinates**: {response.Latitude():.4f}°N, {response.Longitude():.4f}°E")
    output.append(f"- **Elevation**: {response.Elevation()} m above sea level")
    output.append(f"- **Analysis Period**: {days} days ({daily_agg['day'].iloc[0]} to {daily_agg['day'].iloc[-1]})\n")

    # Deep vertical profiles advisories
    output.append("## 🪴 Root Zone Agricultural Advisories\n")

    # 1. Surface Zone (0-7cm) - Germination suitability
    output.append("### 🟫 Surface Zone (0-7 cm)")
    output.append("   *Crucial for seed sowing, germination, and shallow-root crops (e.g., lettuce, onions).*")
    if avg_t1 < 10.0:
        output.append(f"> [!WARNING]\n> **Cold Surface Soil**: Average temperature is {avg_t1:.1f}°C. Warm-season seeds will rot or remain dormant. Delay sowing until soil temperatures exceed 10-15°C.")
    elif avg_t1 < 15.0:
        output.append(f"> [!IMPORTANT]\n> **Cool Surface Soil**: Average temperature is {avg_t1:.1f}°C. Cool-season crops (e.g. peas, spinach) can germinate, but warm-season crops will have delayed emergence.")
    else:
        output.append(f"> [!TIP]\n> **Warm Surface Soil**: Average temperature is {avg_t1:.1f}°C. Favorable for swift germination of most warm-season crops (tomatoes, squash, corn).")

    if avg_m1 < 0.15:
        output.append(f"> [!CAUTION]\n> **Dry Surface**: Moisture level is low ({avg_m1:.3f} m³/m³). Seeds will fail to imbibe water and germinate. Immediate irrigation is necessary after planting.")
    elif avg_m1 > 0.45:
        output.append(f"> [!WARNING]\n> **Waterlogged Surface**: Moisture is extremely high ({avg_m1:.3f} m³/m³). High risk of seed decay, damping-off fungus, and compaction. Avoid field traffic.")
    else:
        output.append(f"> [!TIP]\n> **Optimal Surface Moisture**: Moisture level is ideal ({avg_m1:.3f} m³/m³). Highly favorable for root emergence.")

    # 2. Medium Root Zone (7-28 cm) - Active plant growth
    output.append("\n### 🥕 Active Root Zone (7-28 cm)")
    output.append("   *Primary absorption zone for mature vegetable roots, grains, and small shrubs.*")
    if avg_m2 < 0.15:
        output.append(f"> [!WARNING]\n> **Active Root Drought**: Volumetric soil moisture at 7-28 cm is {avg_m2:.3f} m³/m³. The plants' active root zone is entering drought stress, which will lead to wilting and reduced yield. Deep irrigation is highly recommended.")
    elif avg_m2 > 0.45:
        output.append(f"> [!CAUTION]\n> **Saturated Subsurface**: Subsurface moisture is {avg_m2:.3f} m³/m³. Soil pores are saturated, causing lack of oxygen (anoxia) to plant roots. Root death and nutrient uptake blockages may occur.")
    else:
        output.append(f"> [!TIP]\n> **Healthy Subsurface Moisture**: Moisture level is healthy ({avg_m2:.3f} m³/m³). Promotes healthy crop growth and strong transpiration.")

    # 3. Deep Root Zone (28-100 cm)
    output.append("\n### 🌳 Deep Root Zone (28-100 cm)")
    output.append("   *Key water source for deep-rooted crops, grapevines, and fruit trees.*")
    if avg_m3 < 0.15:
        output.append(f"> [!IMPORTANT]\n> **Deep Soil Dryness**: Moisture is low ({avg_m3:.3f} m³/m³). Deep root systems and orchards are exhausting their reserve moisture. Deep-soak irrigation may be necessary.")
    else:
        output.append(f"> [!TIP]\n> **Stable Deep Moisture**: Moisture is stable ({avg_m3:.3f} m³/m³). Provides a resilient reserve buffer against dry hot spells.")

    output.append("")

    # Vertical Profile Table
    output.append("## 📊 Soil Depth Profile Statistics (Average over period)\n")
    output.append("| Layer Depth | Average Temperature (°C) | Average Volumetric Moisture (m³/m³) | Suitability Status |")
    output.append("| --- | --- | --- | --- |")
    
    def get_status(temp, moist):
        if moist < 0.15:
            return "🔴 Dry (Drought)"
        elif moist > 0.45:
            return "🔵 Saturated (Anoxic)"
        elif temp < 10.0:
            return "🟡 Cold"
        else:
            return "🟢 Ideal"

    output.append(f"| Level 1: 0 - 7 cm | {avg_t1:.1f}°C | {avg_m1:.3f} m³/m³ | {get_status(avg_t1, avg_m1)} |")
    output.append(f"| Level 2: 7 - 28 cm | {avg_t2:.1f}°C | {avg_m2:.3f} m³/m³ | {get_status(avg_t2, avg_m2)} |")
    output.append(f"| Level 3: 28 - 100 cm | {avg_t3:.1f}°C | {avg_m3:.3f} m³/m³ | {get_status(avg_t3, avg_m3)} |")
    output.append(f"| Level 4: 100 - 255 cm | {avg_t4:.1f}°C | {avg_m4:.3f} m³/m³ | {get_status(avg_t4, avg_m4)} |")
    output.append("")

    # Daily soil forecast table
    output.append("## 📅 Daily Soil Profile Forecast\n")
    output.append("| Date | Depth 0-7cm (T / M) | Depth 7-28cm (T / M) | Depth 28-100cm (T / M) | Depth 100-255cm (T / M) |")
    output.append("| --- | --- | --- | --- | --- |")
    for _, row in daily_agg.iterrows():
        date_str = row['day'].strftime('%Y-%m-%d')
        output.append(
            f"| {date_str} "
            f"| {row['t1']:.1f}°C / {row['m1']:.3f} "
            f"| {row['t2']:.1f}°C / {row['m2']:.3f} "
            f"| {row['t3']:.1f}°C / {row['m3']:.3f} "
            f"| {row['t4']:.1f}°C / {row['m4']:.3f} |"
        )
    output.append("")
    return "\n".join(output)

@mcp.tool
def get_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str
) -> str:
    """
    Retrieve historical weather data and agricultural summary statistics for a past timeframe.

    Args:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        start_date (str): Start date of the range (YYYY-MM-DD).
        end_date (str): End date of the range (YYYY-MM-DD).
    """
    # Simple date format validation
    import datetime
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD (e.g. '2025-05-01')."

    if start_dt > end_dt:
        return "Error: start_date must be before or equal to end_date."
        
    # Open-Meteo archive starts from 1940 but typically ends 2-3 days ago
    today = datetime.date.today()
    if end_dt >= today - datetime.timedelta(days=1):
        return f"Error: Historical archive data is typically available up to 2 days ago. Please select an end_date prior to {today - datetime.timedelta(days=1)}."

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }

    logger.info(f"Querying historical weather for ({latitude}, {longitude}) from {start_date} to {end_date}...")
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        logger.error(f"Open-Meteo Historical API query failed: {str(e)}")
        return f"Error: Failed to fetch historical data from Open-Meteo API: {str(e)}"

    daily = response.Daily()
    utc_offset = response.UtcOffsetSeconds()

    try:
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "temp_max": daily.Variables(0).ValuesAsNumpy(),
            "temp_min": daily.Variables(1).ValuesAsNumpy(),
            "precipitation": daily.Variables(2).ValuesAsNumpy(),
            "wind_max": daily.Variables(3).ValuesAsNumpy()
        }
        df = pd.DataFrame(data=daily_data)
        # Shift timezone
        df["date_local"] = df["date"] + pd.to_timedelta(utc_offset, unit="s")
        df["date_str"] = df["date_local"].dt.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Error compiling historical pandas data: {str(e)}")
        return f"Error: Failed to process historical data: {str(e)}"

    # Metrics calculation
    total_precip = df["precipitation"].sum()
    avg_max_temp = df["temp_max"].mean()
    avg_min_temp = df["temp_min"].mean()
    absolute_max = df["temp_max"].max()
    absolute_min = df["temp_min"].min()
    
    # Frost days (min temp < 0)
    frost_days = len(df[df["temp_min"] < 0])
    
    # Heat stress days (max temp > 30)
    heat_days = len(df[df["temp_max"] > 30])
    
    output = []
    output.append(f"# AgroConnectMCP Historical Weather Assessment Report\n")
    output.append(f"- **Coordinates**: {response.Latitude():.4f}°N, {response.Longitude():.4f}°E")
    output.append(f"- **Elevation**: {response.Elevation()} m above sea level")
    output.append(f"- **Historical Range**: {start_date} to {end_date} ({len(df)} days)\n")

    output.append("## 📊 Historical Agriculture Metrics Summary\n")
    
    # Cumulative Rainfall Alert
    if total_precip == 0:
        output.append("> [!WARNING]")
        output.append(f"> **Zero Rain Recorded**: No precipitation occurred during this {len(df)}-day window. Extreme drought pressure could have occurred without irrigation.")
    elif total_precip < len(df) * 1.0: # less than 1mm/day average
        output.append("> [!IMPORTANT]")
        output.append(f"> **Low Rainfall**: Cumulative precipitation was {total_precip:.1f} mm. Sub-optimal water input from natural rainfall.")
    else:
        output.append("> [!TIP]")
        output.append(f"> **Adequate Moisture**: Cumulative precipitation was {total_precip:.1f} mm. Favorable water replenishment for field soils.")

    # Extreme Temperatures
    if frost_days > 0:
        output.append("> [!CAUTION]")
        output.append(f"> **Frost Damage Risk**: {frost_days} frost days (Temp < 0°C) were recorded, with a minimum low of {absolute_min:.1f}°C. Severe threat to budding orchards, tender shoots, and young crops.")
        
    if heat_days > 0:
        output.append("> [!WARNING]")
        output.append(f"> **Heat Stress Stressors**: {heat_days} days exceeding 30.0°C were recorded, peaking at {absolute_max:.1f}°C. High crop transpiration and potential flower abortion in sensitive crops.")

    output.append(f"\n### Climatological Summary Stat Cards")
    output.append(f"- **Cumulative Rainfall**: {total_precip:.1f} mm")
    output.append(f"- **Mean Max Temp**: {avg_max_temp:.1f}°C (Peak: {absolute_max:.1f}°C)")
    output.append(f"- **Mean Min Temp**: {avg_min_temp:.1f}°C (Low: {absolute_min:.1f}°C)")
    output.append(f"- **Frost Days (<0°C)**: {frost_days}")
    output.append(f"- **Heat Stress Days (>30°C)**: {heat_days}")
    output.append("")

    # Daily weather log table
    output.append("## 📅 Daily Historical Weather Log\n")
    output.append("| Date | Max Temp (°C) | Min Temp (°C) | Precipitation (mm) | Max Wind (km/h) |")
    output.append("| --- | --- | --- | --- | --- |")
    for _, row in df.iterrows():
        output.append(
            f"| {row['date_str']} "
            f"| {row['temp_max']:.1f}°C "
            f"| {row['temp_min']:.1f}°C "
            f"| {row['precipitation']:.1f} "
            f"| {row['wind_max']:.1f} |"
        )
    output.append("")
    return "\n".join(output)

if __name__ == "__main__":
    logger.info("Starting AgroConnectMCP Server...")
    mcp.run()


