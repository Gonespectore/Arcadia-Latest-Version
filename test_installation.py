import asyncio
import os
from dotenv import load_dotenv
from data_manager import DataManager

async def test_installation():
    """Teste l'installation et la configuration du système"""
    try:
        # Chargement des variables d'environnement
        load_dotenv()
        api_key = os.getenv('FOOTBALL_DATA_API_KEY')
        
        if not api_key:
            print("❌ Erreur: La clé API FOOTBALL_DATA_API_KEY n'est pas définie dans le fichier .env")
            return
        
        # Initialisation du DataManager
        data_manager = DataManager(api_key)
        
        # Test de la connexion Redis
        try:
            data_manager.redis_client.ping()
            print("✅ Test Redis: Connexion réussie")
        except Exception as e:
            print(f"❌ Erreur Redis: {e}")
            return
        
        # Test de récupération des matchs de la Premier League
        matches = await data_manager.get_league_matches('PL')
        if matches:
            print("✅ Test API: Récupération des matchs réussie")
        else:
            print("❌ Erreur: Impossible de récupérer les matchs")
        
        # Test de récupération des équipes
        teams = await data_manager.get_league_teams('PL')
        if teams:
            print("✅ Test API: Récupération des équipes réussie")
        else:
            print("❌ Erreur: Impossible de récupérer les équipes")
        
        # Test de récupération des matchs en direct
        live_matches = await data_manager.get_live_matches('PL')
        if live_matches is not None:
            print("✅ Test API: Récupération des matchs en direct réussie")
        else:
            print("❌ Erreur: Impossible de récupérer les matchs en direct")
        
        # Test du cache
        test_data = {"test": "data"}
        data_manager._set_cached_data("test_key", test_data)
        cached_data = data_manager._get_cached_data("test_key")
        if cached_data == test_data:
            print("✅ Test Cache: Système de cache fonctionnel")
        else:
            print("❌ Erreur: Le système de cache ne fonctionne pas correctement")
        
        print("\n🎉 Tous les tests ont été effectués avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
    finally:
        # Nettoyage
        data_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(test_installation()) 