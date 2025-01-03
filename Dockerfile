# Sử dụng image Python chính thức từ Docker Hub
FROM python:3.12

# Cài đặt thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy yêu cầu cài đặt vào container
COPY requirements.txt .
RUN pip install --upgrade pip
# Cài đặt các thư viện Python cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở cổng 5000 để chạy Flask app
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app/app.py"]
