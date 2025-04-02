import logging
import os
from datetime import datetime

# Création du dossier logs s'il n'existe pas
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Configuration des loggers spécifiques
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger 