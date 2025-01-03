#!/bin/bash

# Bước 1: Xây dựng Docker image
echo "Building Docker image..."
docker build -t nlp-spell-correction .

# Bước 2: Chạy Docker container
echo "Running Docker container..."
docker run -p 5001:5001 nlp-spell-correction

