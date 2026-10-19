FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install git for public repository cloning
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Force the CPU-only installation of PyTorch FIRST
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements (it will detect torch is already installed)
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]