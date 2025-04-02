# Bot de Prédiction de Matchs de Football

Un bot Telegram qui prédit les résultats des matchs de football en utilisant l'IA et l'analyse de données.

## Fonctionnalités

- Prédiction de résultats de matchs
- Statistiques détaillées des équipes
- Analyse en temps réel
- Support des principales ligues européennes
- Cache multi-niveaux pour des performances optimales

## Prérequis

- Python 3.11+
- Redis
- Compte Telegram
- Clé API Football-Data.org

## Installation Locale

1. Cloner le repository :
```bash
git clone [votre-repo]
cd Bot-prediction-matchs
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement dans `.env` :
```
TELEGRAM_BOT_TOKEN=votre_token_telegram
FOOTBALL_DATA_API_KEY=votre_clé_api
```

5. Lancer le bot :
```bash
python bot.py
```

## Déploiement sur Heroku

1. Installer Heroku CLI et se connecter :
```bash
heroku login
```

2. Créer une nouvelle application Heroku :
```bash
heroku create votre-app-name
```

3. Ajouter Redis :
```bash
heroku addons:create heroku-redis:hobby-dev
```

4. Configurer les variables d'environnement :
```bash
heroku config:set TELEGRAM_BOT_TOKEN=votre_token_telegram
heroku config:set FOOTBALL_DATA_API_KEY=votre_clé_api
```

5. Déployer :
```bash
git push heroku main
```

6. Activer le worker :
```bash
heroku ps:scale worker=1
```

## Commandes du Bot

- `/start` - Démarrer le bot
- `/predire_match` - Prédire un match
- `/stats` - Voir les statistiques
- `/aide` - Obtenir de l'aide

## Support

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub. 