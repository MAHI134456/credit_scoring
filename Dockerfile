# Use a lightweight Python image
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app code and training script into the container
COPY . .

# Remove old mlruns directory to avoid Windows path issues
RUN rm -rf /app/mlruns

# Run training script to register the model with Linux paths
RUN python src/train.py

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
