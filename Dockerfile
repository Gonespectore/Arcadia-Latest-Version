FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers du projet
COPY requirements.txt .
COPY . .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Commande de démarrage
CMD ["python", "bot.py"] 