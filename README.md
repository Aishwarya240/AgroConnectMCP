# AgroConnectMCP
Version: 1.0.0

AgroConnectMCP is a production-ready Model Context Protocol (MCP) server designed to bring advanced agricultural intelligence to large language models. It provides a comprehensive suite of tools for crop disease diagnosis, weather forecasting, crop and fertilizer recommendations, and agricultural market news.

## Features

AgroConnectMCP exposes a rich set of capabilities via the Model Context Protocol:

*   **Plant Disease Diagnosis:** Identifies plant diseases and pests from images (URL, local path, or base64) and provides detailed treatment suggestions.
*   **Crop Recommendation:** Recommends the optimal crop based on soil nutrient levels (Nitrogen, Phosphorus, Potassium) and environmental factors (Temperature, Humidity, pH, Rainfall) using a trained Random Forest model.
*   **Fertilizer Recommendation:** Suggests appropriate fertilizers based on soil conditions, crop type, environmental factors, and existing nutrient levels.
*   **Weather Intelligence:** Comprehensive weather data powered by Open-Meteo, including:
    *   Current conditions and 14-day forecasts.
    *   Detailed metrics: Temperature, Humidity, Precipitation, Wind, Pressure, Cloud Cover, Visibility, and UV Index.
    *   Soil metrics: Soil temperature and moisture at various depths.
    *   Evapotranspiration and WMO weather codes.
    *   Sunrise and sunset times.
*   **Historical Weather:** Retrieves past weather data for specific locations and date ranges.
*   **River Discharge:** Provides flood risk assessment data via river discharge forecasts.
*   **Agriculture Market News:** Fetches the latest region-specific agricultural market news and crop prices.

## Prerequisites

*   Python 3.12 or higher (if running locally)
*   Docker (if running via container)
*   Kindwise API Key for Crop Health features

### Obtaining a Kindwise API Key

The plant disease diagnosis tool relies on the Kindwise Crop Health API. To use this feature:

1.  Visit the Kindwise website: https://www.kindwise.com/crop-health
2.  Sign up for an account and navigate to the API section.
3.  Generate a new API key for the Crop Health API.
4.  Keep this key secure. It will be passed to the MCP server as an environment variable (`KINDWISE_API_KEY`).

## Installation and Usage

AgroConnectMCP can be run directly via Python or within a Docker container.

### Option 1: Running with Python (Local Virtual Environment)

This approach is best for development or if you prefer running services directly on your host machine.

1.  **Clone the repository and navigate to the directory:**
    ```bash
    git clone <repository_url>
    cd AgroConnectMCP
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the Environment:**
    Create a `.env` file in the project root and add your API key:
    ```env
    KINDWISE_API_KEY=your_actual_api_key_here
    ```

5.  **Run the Server:**
    The server uses STDIO transport, which is standard for MCP.
    ```bash
    python server.py
    ```
    *Note: Running this directly in a terminal will start the server waiting for JSON-RPC messages via standard input.*

### Option 2: Running with Docker (Production Recommended)

Docker provides an isolated, consistent environment, making it ideal for production deployments.

1.  **Build the Docker image:**
    ```bash
    docker build -t agroconnectmcp:1.0.0 .
    ```

2.  **Run the container:**
    Pass the Kindwise API key as an environment variable. The container is configured to run the server in STDIO mode.
    ```bash
    docker run -i --rm -e KINDWISE_API_KEY="your_api_key_here" agroconnectmcp:1.0.0
    ```

## Testing the Server

You can inspect and test the available tools using the official MCP Inspector. This provides a web interface to interact with the server.

**Testing the local Python installation:**
```bash
npx @modelcontextprotocol/inspector python server.py
```

**Testing the Docker container:**
```bash
npx @modelcontextprotocol/inspector docker run -i --rm -e KINDWISE_API_KEY="your_api_key_here" agroconnectmcp:1.0.0
```

## Configuring MCP Clients

To use AgroConnectMCP with an MCP-compatible client (like Claude Desktop or LM Studio), configure the client to start the server.

### Claude Desktop Integration

Claude Desktop requires you to edit its configuration file.

**Configuration File Location:**
*   **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
*   **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the following to the configuration file:

**Option A: Using Local Python**
```json
{
  "mcpServers": {
    "AgroConnectMCP": {
      "command": "/absolute/path/to/AgroConnectMCP/.venv/bin/python",
      "args": ["/absolute/path/to/AgroConnectMCP/server.py"],
      "env": {
        "KINDWISE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Option B: Using Docker**
```json
{
  "mcpServers": {
    "AgroConnectMCP": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "KINDWISE_API_KEY=your_api_key_here",
        "agroconnectmcp:1.0.0"
      ]
    }
  }
}
```

After updating the configuration, restart Claude Desktop for the changes to take effect.

### LM Studio Integration

LM Studio supports MCP servers natively. You can add the server via the UI or by editing the LM Studio configuration file.

**Adding via Configuration File:**
Edit your `mcp_config.json` (typically located in `~/.cache/lm-studio/mcp/` or your system's equivalent directory) or add it directly through the LM Studio developer settings.

**Option A: Using Local Python**
```json
{
  "mcpServers": {
    "AgroConnectMCP": {
      "command": "/absolute/path/to/AgroConnectMCP/.venv/bin/python",
      "args": ["/absolute/path/to/AgroConnectMCP/server.py"],
      "env": {
        "KINDWISE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Option B: Using Docker**
```json
{
  "mcpServers": {
    "AgroConnectMCP": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "KINDWISE_API_KEY=your_api_key_here",
        "agroconnectmcp:1.0.0"
      ]
    }
  }
}
```

Once added, the server will connect automatically, and the agricultural tools will be available in your LM Studio chat sessions.

## Architecture Details

*   **Framework:** Built using `FastMCP` for rapid and structured MCP server development.
*   **Machine Learning:** Utilizes serialized `scikit-learn` and `xgboost` models for local inference, ensuring fast responses for crop and fertilizer recommendations without external API dependencies.
*   **External APIs:** Integrates with Open-Meteo for comprehensive weather data and DuckDuckGo for market news.

## Support

For issues, feature requests, or contributions, please refer to the project's issue tracker.
