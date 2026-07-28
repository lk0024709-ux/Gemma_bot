# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level deps required by some Python packages (if any).
# Keep this minimal; add packages here only when required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose default port (optional metadata)
EXPOSE 10000

# Run the FastAPI app with Uvicorn and bind to 0.0.0.0 and the $PORT env var Render provides.
# Use sh -c so the shell expands the ${PORT:-10000} default.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
