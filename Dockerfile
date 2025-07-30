FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file for the FastAPI app
COPY app/requirements.txt .

# Install the application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ .

# Expose the port the app runs on
EXPOSE 8000

# Run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
