FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Data_sets/ ./Data_sets/
COPY config/ ./config/
COPY Scripts/ ./Scripts/

RUN mkdir -p models reports
