import os
import base64
import mimetypes
import logging
from typing import Optional
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgroConnectMCP")

# Load environment variables from .env file
load_dotenv()

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

if __name__ == "__main__":
    logger.info("Starting AgroConnectMCP Server...")
    mcp.run()

