# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory structure
COPY app/ ./app/
COPY configs/ ./configs/

# Copy environment sample file (user should provide actual .env)
COPY .env.sample .

# Copy other necessary files
COPY .python-version .
COPY demo_streaming_wor* .
COPY package-lock.json .
COPY README.md .

# Expose the backend port (FastAPI default is 8000)
EXPOSE 8000

# Run the FastAPI app
# Adjust the module path based on where your FastAPI app instance is located
# If main.py is in the root, use: main:app
# If main.py is in app/, use: app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]