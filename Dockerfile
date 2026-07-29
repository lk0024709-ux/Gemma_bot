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

# Run the Telegram bot directly with polling
CMD ["python", "bot_handler.py"]
