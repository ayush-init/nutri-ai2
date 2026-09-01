FROM python:3.12-slim

WORKDIR /app

# Install system libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY runs/ /app/runs/
COPY data/ /app/data/

ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
