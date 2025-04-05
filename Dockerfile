FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Unzip the model.zip
RUN apt-get update && apt-get install -y unzip && \
    unzip mental_health_model.zip -d ./mental_health_model/ && \
    rm mental_health_model.zip

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
