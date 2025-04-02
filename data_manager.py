import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from functools import lru_cache
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
from typing import Dict, List, Optional, Tuple, Any
import pickle
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.football-data.org/v4"
        self.openliga_url = "https://api.openligadb.de"
        self.thesportsdb_url = "https://www.thesportsdb.com/api/v1/json"
        self.headers = {
            'X-Auth-Token': api_key,
            'User-Agent': 'Football-Prediction-Bot/1.0'
        }
        
        # Configuration des ligues
        self.leagues = {
            'PL': {
                'name': 'Premier League',
                'country': 'England',
                'source': 'football-data.org',
                'id': 'PL',
                'update_interval': 5  # minutes
            },
            'PD': {
                'name': 'La Liga',
                'country': 'Spain',
                'source': 'football-data.org',
                'id': 'PD',
                'update_interval': 5
            },
            'BL1': {
                'name': 'Bundesliga',
                'country': 'Germany',
                'source': 'openligadb',
                'id': 'BL1',
                'update_interval': 3
            },
            'SA': {
                'name': 'Serie A',
                'country': 'Italy',
                'source': 'football-data.org',
                'id': 'SA',
                'update_interval': 5
            },
            'FL1': {
                'name': 'Ligue 1',
                'country': 'France',
                'source': 'football-data.org',
                'id': 'FL1',
                'update_interval': 5
            }
        }
        
        # Configuration du cache Redis
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Cache en mémoire avec TTL
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
        self.teams_cache = {}
        self.leagues_cache = {}
        self.stats_cache = {}
        self.matches_cache = {}
        
        # Création des dossiers nécessaires
        self.cache_dir = "cache"
        self.models_dir = "models"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialisation du pool de threads
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Configuration des limites de l'API
        self.rate_limits = {
            'football-data.org': {'requests': 10, 'period': 60},
            'openligadb': {'requests': 30, 'period': 60},
            'thesportsdb': {'requests': 20, 'period': 60}
        }
        self.last_request_times = {}
        self.request_counts = {}
        
        # Initialisation du scheduler pour les mises à jour
        self.scheduler = AsyncIOScheduler()
        self._setup_update_scheduler()

    def _setup_update_scheduler(self):
        """Configure le scheduler pour les mises à jour automatiques"""
        for league_code, league_info in self.leagues.items():
            self.scheduler.add_job(
                self._update_league_data,
                'interval',
                minutes=league_info['update_interval'],
                args=[league_code],
                id=f'update_{league_code}'
            )
        self.scheduler.start()

    async def _update_league_data(self, league_code: str):
        """Met à jour les données d'une ligue"""
        try:
            logger.info(f"Mise à jour des données pour la ligue {league_code}")
            
            # Mise à jour des matchs
            matches = await self.get_league_matches(league_code)
            if matches:
                self._update_matches_cache(league_code, matches)
            
            # Mise à jour des équipes
            teams = await self.get_league_teams(league_code)
            if teams:
                self._update_teams_cache(league_code, teams)
            
            # Mise à jour des statistiques
            for team in teams:
                stats = await self.get_team_stats(team['id'])
                if stats:
                    self._update_team_stats_cache(team['id'], stats)
            
            logger.info(f"Mise à jour terminée pour la ligue {league_code}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la ligue {league_code}: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_api_request(self, url: str, source: str, params: Dict = None) -> Optional[Dict]:
        """Fait une requête API avec gestion du rate limiting et retry"""
        current_time = datetime.now()
        rate_limit = self.rate_limits[source]
        
        if source not in self.last_request_times:
            self.last_request_times[source] = current_time
            self.request_counts[source] = 0
        
        if (current_time - self.last_request_times[source]).seconds < rate_limit['period']:
            if self.request_counts[source] >= rate_limit['requests']:
                wait_time = rate_limit['period'] - (current_time - self.last_request_times[source]).seconds
                logger.info(f"Rate limit atteint pour {source}, attente de {wait_time} secondes")
                await asyncio.sleep(wait_time)
                self.request_counts[source] = 0
                self.last_request_times[source] = datetime.now()
        
        self.request_counts[source] += 1
        self.last_request_times[source] = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        logger.warning(f"Rate limit atteint pour {source}")
                        await asyncio.sleep(rate_limit['period'])
                        return await self._make_api_request(url, source, params)
                    else:
                        logger.error(f"Erreur API {source}: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Erreur lors de la requête API {source}: {e}")
                return None

    def _update_matches_cache(self, league_code: str, matches: List[Dict]):
        """Met à jour le cache des matchs"""
        cache_key = f"league_matches_{league_code}"
        self.matches_cache[cache_key] = (matches, datetime.now())
        self._set_cached_data(cache_key, matches)

    def _update_teams_cache(self, league_code: str, teams: List[Dict]):
        """Met à jour le cache des équipes"""
        cache_key = f"league_teams_{league_code}"
        self.teams_cache[cache_key] = (teams, datetime.now())
        self._set_cached_data(cache_key, teams)

    def _update_team_stats_cache(self, team_id: int, stats: Dict):
        """Met à jour le cache des statistiques d'équipe"""
        cache_key = f"team_stats_{team_id}"
        self.stats_cache[cache_key] = (stats, datetime.now())
        self._set_cached_data(cache_key, stats)

    async def get_live_matches(self, league_id: str = None) -> List[Dict]:
        """Récupère les matchs en direct"""
        try:
            # Vérification du cache
            cache_key = f"live_matches_{league_id if league_id else 'all'}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # Construction de l'URL
            url = f"{self.base_url}/matches"
            params = {'status': 'LIVE'}
            if league_id:
                params['competitions'] = league_id

            # Récupération des données
            data = await self._make_api_request(url, params)
            if not data or 'matches' not in data:
                return []

            # Traitement des données
            live_matches = []
            for match in data['matches']:
                match_info = {
                    'id': match['id'],
                    'competition': match['competition']['name'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'score': {
                        'home': match['score']['fullTime']['home'],
                        'away': match['score']['fullTime']['away']
                    },
                    'minute': match['minute'],
                    'status': match['status']
                }
                live_matches.append(match_info)

            # Mise en cache
            self._set_cached_data(cache_key, live_matches, ttl=300)  # Cache de 5 minutes pour les matchs en direct
            return live_matches

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des matchs en direct: {e}")
            return []

    async def get_upcoming_matches(self, hours: int = 24) -> List[Dict]:
        """Récupère les prochains matchs dans les heures à venir"""
        upcoming_matches = []
        try:
            for league_code in self.leagues:
                matches = await self.get_league_matches(league_code)
                now = datetime.now()
                future = now + timedelta(hours=hours)
                
                for match in matches:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d %H:%M:%S')
                    if now <= match_date <= future:
                        upcoming_matches.append(match)
            
            return upcoming_matches
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prochains matchs: {e}")
            return []

    async def get_recent_matches(self, days: int = 7) -> List[Dict]:
        """Récupère les matchs récents"""
        recent_matches = []
        try:
            for league_code in self.leagues:
                matches = await self.get_league_matches(league_code)
                now = datetime.now()
                past = now - timedelta(days=days)
                
                for match in matches:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d %H:%M:%S')
                    if past <= match_date <= now:
                        recent_matches.append(match)
            
            return recent_matches
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des matchs récents: {e}")
            return []

    def shutdown(self):
        """Arrête le scheduler et nettoie les ressources"""
        self.scheduler.shutdown()
        self.thread_pool.shutdown(wait=True)
        self.redis_client.close()

    def _get_redis_cache(self, key: str) -> Optional[Dict]:
        """Récupère les données du cache Redis"""
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Erreur Redis: {e}")
        return None

    def _set_redis_cache(self, key: str, data: Dict, ttl: int = 3600):
        """Stocke les données dans Redis"""
        try:
            self.redis_client.setex(
                key,
                ttl,
                pickle.dumps(data)
            )
        except Exception as e:
            logger.error(f"Erreur Redis: {e}")

    def _get_cached_data(self, key: str) -> Optional[Dict]:
        """Récupère les données du cache multi-niveaux"""
        # 1. Vérification du cache Redis
        redis_data = self._get_redis_cache(key)
        if redis_data:
            return redis_data
        
        # 2. Vérification du cache en mémoire
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        
        # 3. Vérification du cache fichier
        cached_data = self._load_from_file_cache(key)
        if cached_data:
            self.cache[key] = (cached_data, datetime.now())
            self._set_redis_cache(key, cached_data)
            return cached_data
        
        return None

    def _set_cached_data(self, key: str, data: Dict):
        """Stocke les données dans le cache multi-niveaux"""
        # 1. Mise en cache Redis
        self._set_redis_cache(key, data)
        
        # 2. Mise en cache en mémoire
        self.cache[key] = (data, datetime.now())
        
        # 3. Mise en cache fichier
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def _load_from_file_cache(self, key: str) -> Optional[Dict]:
        """Charge les données depuis le cache fichier"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement du cache fichier {key}: {e}")
        return None

    def search_team(self, team_name: str) -> Optional[Dict]:
        """Recherche une équipe par son nom"""
        cache_key = f"team_search_{team_name}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            response = requests.get(
                f"{self.base_url}/teams",
                headers=self.headers,
                params={'name': team_name}
            )
            response.raise_for_status()
            teams = response.json().get('teams', [])
            
            if teams:
                team_data = teams[0]
                self._set_cached_data(cache_key, team_data)
                return team_data
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de l'équipe {team_name}: {e}")
            return None

    def get_team_matches(self, team_id: int, limit: int = 10) -> List[Dict]:
        """Récupère les derniers matchs d'une équipe"""
        cache_key = f"team_matches_{team_id}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            response = requests.get(
                f"{self.base_url}/teams/{team_id}/matches",
                headers=self.headers,
                params={'limit': limit}
            )
            response.raise_for_status()
            matches = response.json().get('matches', [])
            
            self._set_cached_data(cache_key, matches)
            return matches
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des matchs de l'équipe {team_id}: {e}")
            return []

    @lru_cache(maxsize=50)
    async def get_league_matches(self, league_code: str, season: str = None) -> List[Dict]:
        """Récupère les matchs d'une ligue depuis la source appropriée"""
        cache_key = f"league_matches_{league_code}_{season}"
        
        # Vérification du cache
        if cache_key in self.matches_cache:
            data, timestamp = self.matches_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        
        league_info = self.leagues.get(league_code)
        if not league_info:
            logger.error(f"Ligue non trouvée: {league_code}")
            return []
        
        try:
            if league_info['source'] == 'football-data.org':
                params = {'season': season} if season else {}
                response = await self._make_api_request(
                    f"{self.base_url}/competitions/{league_code}/matches",
                    'football-data.org',
                    params
                )
                if response:
                    matches_data = response['matches']
            elif league_info['source'] == 'openligadb':
                response = await self._make_api_request(
                    f"{self.openliga_url}/getmatchdata/{league_code}",
                    'openligadb'
                )
                if response:
                    matches_data = self._format_openliga_matches(response)
            else:
                logger.error(f"Source de données non supportée: {league_info['source']}")
                return []
            
            # Mise en cache
            self.matches_cache[cache_key] = (matches_data, datetime.now())
            self._set_cached_data(cache_key, matches_data)
            
            return matches_data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des matchs de la ligue {league_code}: {e}")
            return []

    def _format_openliga_matches(self, data: List[Dict]) -> List[Dict]:
        """Formate les données de matchs d'OpenLigaDB"""
        formatted_matches = []
        for match in data:
            formatted_match = {
                'id': match.get('MatchID'),
                'utcDate': match.get('MatchDateTime'),
                'status': 'FINISHED' if match.get('MatchIsFinished') else 'SCHEDULED',
                'homeTeam': {
                    'id': match.get('Team1', {}).get('TeamId'),
                    'name': match.get('Team1', {}).get('TeamName')
                },
                'awayTeam': {
                    'id': match.get('Team2', {}).get('TeamId'),
                    'name': match.get('Team2', {}).get('TeamName')
                },
                'score': {
                    'halfTime': {
                        'home': match.get('MatchResults', [{}])[0].get('PointsTeam1', 0),
                        'away': match.get('MatchResults', [{}])[0].get('PointsTeam2', 0)
                    },
                    'fullTime': {
                        'home': match.get('MatchResults', [{}])[-1].get('PointsTeam1', 0),
                        'away': match.get('MatchResults', [{}])[-1].get('PointsTeam2', 0)
                    }
                }
            }
            formatted_matches.append(formatted_match)
        return formatted_matches

    @lru_cache(maxsize=10)
    async def get_league_teams(self, league_code: str) -> List[Dict]:
        """Récupère les équipes d'une ligue depuis la source appropriée"""
        cache_key = f"league_teams_{league_code}"
        
        # Vérification du cache
        if cache_key in self.teams_cache:
            data, timestamp = self.teams_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        
        league_info = self.leagues.get(league_code)
        if not league_info:
            logger.error(f"Ligue non trouvée: {league_code}")
            return []
        
        try:
            if league_info['source'] == 'football-data.org':
                response = await self._make_api_request(
                    f"{self.base_url}/competitions/{league_code}/teams",
                    'football-data.org'
                )
                if response:
                    teams_data = response['teams']
            elif league_info['source'] == 'openligadb':
                response = await self._make_api_request(
                    f"{self.openliga_url}/getavailableteams/{league_code}",
                    'openligadb'
                )
                if response:
                    teams_data = self._format_openliga_teams(response)
            else:
                logger.error(f"Source de données non supportée: {league_info['source']}")
                return []
            
            # Mise en cache
            self.teams_cache[cache_key] = (teams_data, datetime.now())
            self._set_cached_data(cache_key, teams_data)
            
            return teams_data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des équipes de la ligue {league_code}: {e}")
            return []

    def _format_openliga_teams(self, data: List[Dict]) -> List[Dict]:
        """Formate les données d'équipes d'OpenLigaDB"""
        formatted_teams = []
        for team in data:
            formatted_team = {
                'id': team.get('TeamId'),
                'name': team.get('TeamName'),
                'shortName': team.get('ShortName'),
                'tla': team.get('TeamIconUrl', '').split('/')[-1].split('.')[0],
                'crestUrl': team.get('TeamIconUrl')
            }
            formatted_teams.append(formatted_team)
        return formatted_teams

    def get_global_stats(self) -> Dict:
        """Récupère les statistiques globales"""
        cache_key = "global_stats"
        
        # Vérification du cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        
        try:
            stats = {
                'total_matches': 0,
                'accuracy': 0.0,
                'leagues_covered': 5,
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Calcul des statistiques globales
            leagues = ['PL', 'BL1', 'SA', 'PD', 'FL1']
            total_matches = 0
            correct_predictions = 0
            
            for league in leagues:
                matches = self.get_league_matches(league)
                total_matches += len(matches)
                
                # Calcul de la précision (exemple simplifié)
                correct_predictions += len(matches) * 0.65  # Estimation
            
            stats['total_matches'] = total_matches
            stats['accuracy'] = (correct_predictions / total_matches * 100) if total_matches > 0 else 0
            
            # Mise en cache
            self.cache[cache_key] = (stats, datetime.now())
            self._set_cached_data(cache_key, stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats globales: {e}")
            return {
                'total_matches': 0,
                'accuracy': 0.0,
                'leagues_covered': 0,
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def prepare_match_data(self, team1: str, team2: str) -> Optional[Tuple[Dict, Dict]]:
        """Prépare les données pour la prédiction d'un match"""
        team1_data = self.search_team(team1)
        team2_data = self.search_team(team2)

        if not team1_data or not team2_data:
            return None

        team1_stats = self.get_team_stats(team1_data['id'])
        team2_stats = self.get_team_stats(team2_data['id'])

        return (team1_stats, team2_stats)

    @lru_cache(maxsize=1000)
    async def get_team_stats(self, team_id: int) -> Dict:
        """Récupère les statistiques d'une équipe avec cache multi-niveaux"""
        cache_key = f"team_stats_{team_id}"
        
        # 1. Vérification du cache en mémoire
        if cache_key in self.stats_cache:
            data, timestamp = self.stats_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        
        # 2. Vérification du cache fichier
        cached_data = self._load_from_file_cache(cache_key)
        if cached_data:
            self.stats_cache[cache_key] = (cached_data, datetime.now())
            return cached_data
        
        try:
            # Récupération parallèle des données
            team_data_task = self._make_api_request(f"{self.base_url}/teams/{team_id}", 'football-data.org')
            matches_task = self._make_api_request(
                f"{self.base_url}/teams/{team_id}/matches",
                'football-data.org',
                params={'limit': 50}
            )
            
            team_data, matches = await asyncio.gather(team_data_task, matches_task)
            
            if not team_data or not matches:
                return {}
            
            # Traitement des données avec parallélisation
            stats = {
                'id': team_id,
                'name': team_data['name'],
                'goals_scored': 0,
                'goals_conceded': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'form': [],
                'ht_goals_scored': 0,
                'ht_goals_conceded': 0,
                'st_goals_scored': 0,
                'st_goals_conceded': 0,
                'recent_form': [],
                'momentum': 0,
                'home_performance': 0,
                'away_performance': 0,
                'clean_sheets': 0,
                'failed_to_score': 0,
                'avg_goals_scored': 0,
                'avg_goals_conceded': 0,
                'last_5_matches': [],
                'advanced_stats': {
                    'xG': 0,  # Expected Goals
                    'xGA': 0,  # Expected Goals Against
                    'possession': 0,  # Possession moyenne
                    'shots_on_target': 0,
                    'shots_conceded': 0,
                    'pass_accuracy': 0,
                    'fouls_committed': 0,
                    'fouls_suffered': 0
                }
            }
            
            # Traitement parallèle des matchs
            loop = asyncio.get_event_loop()
            match_stats = await loop.run_in_executor(
                self.thread_pool,
                self._process_matches,
                matches.get('matches', []),
                team_id
            )
            
            # Mise à jour des statistiques
            stats.update(match_stats)
            
            # Calcul des statistiques avancées
            stats['advanced_stats'] = self._calculate_advanced_stats(matches.get('matches', []), team_id)
            
            # Mise en cache multi-niveaux
            self.stats_cache[cache_key] = (stats, datetime.now())
            self._set_cached_data(cache_key, stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats de l'équipe {team_id}: {e}")
            return {}

    def _process_matches(self, matches: List[Dict], team_id: int) -> Dict:
        """Traite les matchs en parallèle"""
        stats = {
            'goals_scored': 0,
            'goals_conceded': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'form': [],
            'ht_goals_scored': 0,
            'ht_goals_conceded': 0,
            'st_goals_scored': 0,
            'st_goals_conceded': 0,
            'clean_sheets': 0,
            'failed_to_score': 0,
            'home_performance': 0,
            'away_performance': 0,
            'last_5_matches': []
        }
        
        for match in matches:
            if match['status'] == 'FINISHED':
                is_home = match['homeTeam']['id'] == team_id
                goals_for = match['score']['fullTime']['home'] if is_home else match['score']['fullTime']['away']
                goals_against = match['score']['fullTime']['away'] if is_home else match['score']['fullTime']['home']
                ht_goals_for = match['score']['halfTime']['home'] if is_home else match['score']['halfTime']['away']
                ht_goals_against = match['score']['halfTime']['away'] if is_home else match['score']['halfTime']['home']
                
                # Mise à jour des statistiques
                stats['goals_scored'] += goals_for
                stats['goals_conceded'] += goals_against
                stats['ht_goals_scored'] += ht_goals_for
                stats['ht_goals_conceded'] += ht_goals_against
                stats['st_goals_scored'] += (goals_for - ht_goals_for)
                stats['st_goals_conceded'] += (goals_against - ht_goals_against)
                
                if goals_for > goals_against:
                    stats['wins'] += 1
                    stats['form'].append('W')
                elif goals_for == goals_against:
                    stats['draws'] += 1
                    stats['form'].append('D')
                else:
                    stats['losses'] += 1
                    stats['form'].append('L')
                
                if goals_against == 0:
                    stats['clean_sheets'] += 1
                if goals_for == 0:
                    stats['failed_to_score'] += 1
                
                if is_home:
                    stats['home_performance'] += (goals_for - goals_against)
                else:
                    stats['away_performance'] += (goals_for - goals_against)
                
                if len(stats['last_5_matches']) < 5:
                    stats['last_5_matches'].append({
                        'date': match['utcDate'],
                        'opponent': match['awayTeam']['name'] if is_home else match['homeTeam']['name'],
                        'score': f"{goals_for}-{goals_against}",
                        'is_home': is_home
                    })
        
        # Calcul des moyennes
        total_matches = len(matches)
        if total_matches > 0:
            stats['avg_goals_scored'] = stats['goals_scored'] / total_matches
            stats['avg_goals_conceded'] = stats['goals_conceded'] / total_matches
        
        # Calcul du momentum
        recent_matches = stats['form'][-5:] if stats['form'] else []
        momentum = sum(1 if r == 'W' else 0.5 if r == 'D' else 0 for r in recent_matches)
        stats['momentum'] = momentum / 5 if recent_matches else 0
        
        return stats

    def _calculate_advanced_stats(self, matches: List[Dict], team_id: int) -> Dict:
        """Calcule les statistiques avancées"""
        advanced_stats = {
            'xG': 0,
            'xGA': 0,
            'possession': 0,
            'shots_on_target': 0,
            'shots_conceded': 0,
            'pass_accuracy': 0,
            'fouls_committed': 0,
            'fouls_suffered': 0
        }
        
        total_matches = len(matches)
        if total_matches == 0:
            return advanced_stats
        
        for match in matches:
            if match['status'] == 'FINISHED':
                is_home = match['homeTeam']['id'] == team_id
                stats = match.get('stats', {})
                
                # Calcul des statistiques avancées
                for stat in stats:
                    if stat['team']['id'] == team_id:
                        if stat['type'] == 'Shots on Goal':
                            advanced_stats['shots_on_target'] += stat['value']
                        elif stat['type'] == 'Ball Possession':
                            advanced_stats['possession'] += float(stat['value'].strip('%'))
                        elif stat['type'] == 'Pass Accuracy':
                            advanced_stats['pass_accuracy'] += float(stat['value'].strip('%'))
                        elif stat['type'] == 'Fouls':
                            advanced_stats['fouls_committed'] += stat['value']
                    else:
                        if stat['type'] == 'Shots on Goal':
                            advanced_stats['shots_conceded'] += stat['value']
                        elif stat['type'] == 'Fouls':
                            advanced_stats['fouls_suffered'] += stat['value']
        
        # Calcul des moyennes
        advanced_stats['possession'] /= total_matches
        advanced_stats['pass_accuracy'] /= total_matches
        advanced_stats['shots_on_target'] /= total_matches
        advanced_stats['shots_conceded'] /= total_matches
        advanced_stats['fouls_committed'] /= total_matches
        advanced_stats['fouls_suffered'] /= total_matches
        
        return advanced_stats

# Initialisation
data_manager = DataManager(api_key="votre_clé")

# Récupération des matchs en cours
live_matches = await data_manager.get_live_matches()

# Récupération des prochains matchs
upcoming_matches = await data_manager.get_upcoming_matches(hours=24)

# Arrêt propre
data_manager.shutdown() 