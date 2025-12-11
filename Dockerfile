FROM python:3.9-slim

ENV LANG C.UTF-8

# تثبيت متطلبات النظام اللازمة لـ librosa وخدمات الصوت
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY . /

RUN pip install --no-cache-dir -r requirements.txt