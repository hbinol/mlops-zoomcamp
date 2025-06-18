FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements inline (can also be separated to requirements.txt)
RUN pip install --upgrade pip
RUN pip install pandas numpy scikit-learn pyarrow

# Copy script and model file
COPY starter.py .
COPY model.bin .

# Set default command
ENTRYPOINT ["python", "starter.py"]