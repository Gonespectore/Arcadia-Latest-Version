import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from typing import Dict, Tuple, List
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        # Modèles pour les scores à domicile
        self.home_ht_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.home_ft_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Modèles pour les scores à l'extérieur
        self.away_ht_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.away_ft_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "models"
        os.makedirs(self.model_path, exist_ok=True)
        self.cache_duration = timedelta(hours=1)

    def _create_features(self, team_stats: Dict) -> np.ndarray:
        """Crée les features pour le modèle"""
        features = [
            team_stats.get('goals_scored', 0),
            team_stats.get('goals_conceded', 0),
            team_stats.get('wins', 0),
            team_stats.get('draws', 0),
            team_stats.get('losses', 0),
            len(team_stats.get('form', [])),
            sum(1 for x in team_stats.get('form', []) if x == 'W'),
            sum(1 for x in team_stats.get('form', []) if x == 'D'),
            sum(1 for x in team_stats.get('form', []) if x == 'L'),
            team_stats.get('goals_scored', 0) / max(1, len(team_stats.get('form', []))),
            team_stats.get('goals_conceded', 0) / max(1, len(team_stats.get('form', []))),
            # Nouvelles features pour les mi-temps
            team_stats.get('ht_goals_scored', 0),
            team_stats.get('ht_goals_conceded', 0),
            team_stats.get('st_goals_scored', 0),
            team_stats.get('st_goals_conceded', 0)
        ]
        return np.array(features).reshape(1, -1)

    def _calculate_form_metrics(self, form: List[str]) -> Dict:
        """Calcule des métriques avancées sur la forme"""
        if not form:
            return {'form_weight': 0.5, 'momentum': 0, 'consistency': 0}

        # Poids des résultats récents
        weights = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        recent_form = form[-5:] if len(form) > 5 else form
        form_weight = sum(weights[result] for result in recent_form) / len(recent_form)

        # Calcul du momentum (tendance récente)
        momentum = 0
        for i in range(len(recent_form) - 1):
            if recent_form[i] == 'W' and recent_form[i + 1] == 'W':
                momentum += 1
            elif recent_form[i] == 'L' and recent_form[i + 1] == 'L':
                momentum -= 1

        # Calcul de la cohérence (variation des résultats)
        unique_results = set(recent_form)
        consistency = 1 - (len(unique_results) - 1) / 2

        return {
            'form_weight': form_weight,
            'momentum': momentum,
            'consistency': consistency
        }

    def train(self, historical_data: List[Dict]):
        """Entraîne le modèle sur les données historiques"""
        try:
            X_home = []
            X_away = []
            y_home_ht = []
            y_home_ft = []
            y_away_ht = []
            y_away_ft = []

            for match in historical_data:
                home_features = self._create_features(match['home_stats'])
                away_features = self._create_features(match['away_stats'])
                
                X_home.append(home_features[0])
                X_away.append(away_features[0])
                y_home_ht.append(match['home_ht_goals'])
                y_home_ft.append(match['home_ft_goals'])
                y_away_ht.append(match['away_ht_goals'])
                y_away_ft.append(match['away_ft_goals'])

            X_home = np.array(X_home)
            X_away = np.array(X_away)
            y_home_ht = np.array(y_home_ht)
            y_home_ft = np.array(y_home_ft)
            y_away_ht = np.array(y_away_ht)
            y_away_ft = np.array(y_away_ft)

            # Normalisation des features
            X_home_scaled = self.scaler.fit_transform(X_home)
            X_away_scaled = self.scaler.transform(X_away)

            # Entraînement des modèles
            self.home_ht_model.fit(X_home_scaled, y_home_ht)
            self.home_ft_model.fit(X_home_scaled, y_home_ft)
            self.away_ht_model.fit(X_away_scaled, y_away_ht)
            self.away_ft_model.fit(X_away_scaled, y_away_ft)

            # Évaluation des modèles
            home_ht_pred = self.home_ht_model.predict(X_home_scaled)
            home_ft_pred = self.home_ft_model.predict(X_home_scaled)
            away_ht_pred = self.away_ht_model.predict(X_away_scaled)
            away_ft_pred = self.away_ft_model.predict(X_away_scaled)

            # Calcul des métriques
            metrics = {
                'home_ht': {'mse': mean_squared_error(y_home_ht, home_ht_pred),
                           'r2': r2_score(y_home_ht, home_ht_pred)},
                'home_ft': {'mse': mean_squared_error(y_home_ft, home_ft_pred),
                           'r2': r2_score(y_home_ft, home_ft_pred)},
                'away_ht': {'mse': mean_squared_error(y_away_ht, away_ht_pred),
                           'r2': r2_score(y_away_ht, away_ht_pred)},
                'away_ft': {'mse': mean_squared_error(y_away_ft, away_ft_pred),
                           'r2': r2_score(y_away_ft, away_ft_pred)}
            }

            logger.info("Performance des modèles:")
            for model_name, model_metrics in metrics.items():
                logger.info(f"{model_name} - MSE: {model_metrics['mse']:.2f}, R2: {model_metrics['r2']:.2f}")

            self.is_trained = True
            self._save_models()

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            raise

    def _save_models(self):
        """Sauvegarde les modèles entraînés"""
        try:
            joblib.dump(self.home_ht_model, os.path.join(self.model_path, 'home_ht_model.joblib'))
            joblib.dump(self.home_ft_model, os.path.join(self.model_path, 'home_ft_model.joblib'))
            joblib.dump(self.away_ht_model, os.path.join(self.model_path, 'away_ht_model.joblib'))
            joblib.dump(self.away_ft_model, os.path.join(self.model_path, 'away_ft_model.joblib'))
            joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.joblib'))
            logger.info("Modèles sauvegardés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {e}")

    def _load_models(self):
        """Charge les modèles sauvegardés"""
        try:
            self.home_ht_model = joblib.load(os.path.join(self.model_path, 'home_ht_model.joblib'))
            self.home_ft_model = joblib.load(os.path.join(self.model_path, 'home_ft_model.joblib'))
            self.away_ht_model = joblib.load(os.path.join(self.model_path, 'away_ht_model.joblib'))
            self.away_ft_model = joblib.load(os.path.join(self.model_path, 'away_ft_model.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.joblib'))
            self.is_trained = True
            logger.info("Modèles chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")

    def predict_score(self, team1_stats: Dict, team2_stats: Dict) -> Dict[str, Tuple[int, int]]:
        """Prédit les scores des mi-temps et du match complet"""
        try:
            # Création des features
            home_features = self._create_features(team1_stats)
            away_features = self._create_features(team2_stats)

            # Normalisation des features
            home_features_scaled = self.scaler.transform(home_features)
            away_features_scaled = self.scaler.transform(away_features)

            # Prédiction des buts
            home_ht_goals = max(0, round(self.home_ht_model.predict(home_features_scaled)[0]))
            home_ft_goals = max(0, round(self.home_ft_model.predict(home_features_scaled)[0]))
            away_ht_goals = max(0, round(self.away_ht_model.predict(away_features_scaled)[0]))
            away_ft_goals = max(0, round(self.away_ft_model.predict(away_features_scaled)[0]))

            # Ajustement basé sur la forme récente
            home_form = self._calculate_form_metrics(team1_stats.get('form', []))
            away_form = self._calculate_form_metrics(team2_stats.get('form', []))

            # Ajustement des prédictions
            home_ht_goals = int(home_ht_goals * (1 + home_form['momentum'] * 0.1))
            home_ft_goals = int(home_ft_goals * (1 + home_form['momentum'] * 0.1))
            away_ht_goals = int(away_ht_goals * (1 + away_form['momentum'] * 0.1))
            away_ft_goals = int(away_ft_goals * (1 + away_form['momentum'] * 0.1))

            return {
                'ht': (home_ht_goals, away_ht_goals),
                'ft': (home_ft_goals, away_ft_goals)
            }

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return {
                'ht': (0, 0),
                'ft': (0, 0)
            } 