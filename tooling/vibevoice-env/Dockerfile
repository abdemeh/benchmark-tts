FROM python:3.10-slim

WORKDIR /app

# 1. On installe git (pour cloner le repo) et ffmpeg (toujours utile pour l'audio)
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# 2. On copie les dépendances
COPY requirements.txt .

# 3. On installe les librairies Python
RUN pip install --no-cache-dir -r requirements.txt

# 4. On copie le reste du code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]