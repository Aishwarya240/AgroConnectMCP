# AgroConnectMCP - Agriculture MCP Server
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server and ML models
COPY server.py .
COPY Models/ ./Models/

# API key must be provided at runtime via: docker run -e KINDWISE_API_KEY=your_key
ENV KINDWISE_API_KEY=""
ENV PYTHONUNBUFFERED=1

# Run MCP server (STDIO mode)
ENTRYPOINT ["python", "server.py"]
