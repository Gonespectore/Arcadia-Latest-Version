import os
from data_manager import DataManager
from ml_model import MLPredictor
from dotenv import load_dotenv
import logging
from typing import List, Dict
import time

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

def collect_historical_data(data_manager: DataManager, leagues: List[str], seasons: List[str]) -> List[Dict]:
    """Collecte les données historiques des matchs"""
    historical_data = []
    
    for league in leagues:
        for season in seasons:
            try:
                # Récupération des matchs de la ligue pour la saison
                matches = data_manager.get_league_matches(league, season)
                
                for match in matches:
                    if match['status'] == 'FINISHED':
                        # Récupération des statistiques des équipes
                        home_stats = data_manager.get_team_stats(match['homeTeam']['id'])
                        away_stats = data_manager.get_team_stats(match['awayTeam']['id'])
                        
                        historical_data.append({
                            'home_stats': home_stats,
                            'away_stats': away_stats,
                            'home_ht_goals': match['score']['halfTime']['home'],
                            'away_ht_goals': match['score']['halfTime']['away'],
                            'home_ft_goals': match['score']['fullTime']['home'],
                            'away_ft_goals': match['score']['fullTime']['away']
                        })
                
                # Pause pour respecter les limites de l'API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des données pour {league} {season}: {e}")
                continue
    
    return historical_data

def main():
    """Fonction principale"""
    # Configuration
    FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY')
    if not FOOTBALL_API_KEY:
        logger.error("Clé API Football-Data.org non trouvée!")
        return

    # Initialisation
    data_manager = DataManager(FOOTBALL_API_KEY)
    ml_predictor = MLPredictor()

    # Ligues et saisons à collecter
    leagues = ['PL', 'BL1', 'SA', 'PD', 'FL1']  # Premier League, Bundesliga, Serie A, La Liga, Ligue 1
    seasons = ['2021', '2022', '2023']

    # Collecte des données
    logger.info("Début de la collecte des données historiques...")
    historical_data = collect_historical_data(data_manager, leagues, seasons)
    logger.info(f"Données collectées: {len(historical_data)} matchs")

    if not historical_data:
        logger.error("Aucune donnée historique collectée!")
        return

    # Entraînement du modèle
    logger.info("Début de l'entraînement du modèle...")
    try:
        ml_predictor.train(historical_data)
        logger.info("Entraînement terminé avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}")

if __name__ == '__main__':
    main() 