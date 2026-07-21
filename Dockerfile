FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY src src
COPY models models
COPY data/stations.csv data/stations.csv
COPY data/station_baselines.csv data/station_baselines.csv

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
