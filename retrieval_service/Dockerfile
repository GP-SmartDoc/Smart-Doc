FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose for worker)
CMD ["uvicorn", "--app-dir", "src", "retrieval_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
