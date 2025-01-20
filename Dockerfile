# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV EMAIL_ADDRESS=newstracker502@gmail.com
ENV EMAIL_PASSWORD=Uaa24412
ENV RECIPIENT_EMAIL=erdem_meral@hotmail.com

# Run the application
CMD ["python", "-m", "app.monitoring.real_time_monitor"]

