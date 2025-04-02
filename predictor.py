import numpy as np
from typing import Dict, Tuple
import logging
from ml_model import MLPredictor

logger = logging.getLogger(__name__)

class ScorePredictor:
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.ml_predictor._load_models()  # Charge les modèles sauvegardés s'ils existent

    def _calculate_form_weight(self, form: list) -> float:
        """Calcule un poids basé sur la forme récente de l'équipe"""
        if not form:
            return 0.5
        
        weights = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        recent_form = form[-5:] if len(form) > 5 else form
        return sum(weights[result] for result in recent_form) / len(recent_form)

    def predict_score(self, team1_stats: Dict, team2_stats: Dict) -> Dict[str, Tuple[int, int]]:
        """Prédit les scores des mi-temps et du match complet"""
        try:
            # Utilisation du modèle ML pour la prédiction
            predictions = self.ml_predictor.predict_score(team1_stats, team2_stats)

            # Ajustement final basé sur la forme récente
            team1_form = self._calculate_form_weight(team1_stats.get('form', []))
            team2_form = self._calculate_form_weight(team2_stats.get('form', []))

            # Ajustement des prédictions
            for period in ['ht', 'ft']:
                home_goals, away_goals = predictions[period]
                predictions[period] = (
                    int(home_goals * (1 + (team1_form - 0.5) * 0.2)),
                    int(away_goals * (1 + (team2_form - 0.5) * 0.2))
                )

            return predictions

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du score: {e}")
            return {
                'ht': (0, 0),
                'ft': (0, 0)
            }

    def format_prediction(self, team1: str, team2: str, predictions: Dict[str, Tuple[int, int]]) -> str:
        """Formate la prédiction pour l'affichage"""
        ht_score = predictions['ht']
        ft_score = predictions['ft']
        
        return (
            f"🎯 Prédiction pour le match {team1} vs {team2}:\n\n"
            f"Mi-temps: {team1} {ht_score[0]} - {ht_score[1]} {team2}\n"
            f"Score final: {team1} {ft_score[0]} - {ft_score[1]} {team2}"
        ) 