# Use an appropriate base image, e.g., python:3.10-slim
#FROM --platform=linux/amd64 python:3.9-slim
FROM python:3.9-slim

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy your application's requirements and install them
COPY . /app/

RUN pip install -r /app/requirements.txt

EXPOSE 8080

CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--port", "8080"]