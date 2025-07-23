# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Force numpy<2.0.0 first to avoid Evidently conflict
RUN pip install "numpy<2.0.0"

# Then install the rest of the requirements
RUN pip install --upgrade pip && pip install -r requirements.txt


# Copy all backend files into the container
COPY . .

# Expose FastAPI port
EXPOSE 80

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
