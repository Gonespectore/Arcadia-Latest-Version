import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
from data_manager import DataManager
from predictor import ScorePredictor
import asyncio
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Configuration du bot
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY')

# Initialisation des gestionnaires
data_manager = DataManager(FOOTBALL_API_KEY)
score_predictor = ScorePredictor()

# Cache pour les prédictions fréquentes
prediction_cache = {}
CACHE_DURATION = timedelta(hours=1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /start - Accueil et présentation du bot"""
    welcome_message = (
        "👋 Bienvenue sur le Bot de Prédiction de Matchs de Football!\n\n"
        "⚽ Je peux prédire les scores des matchs en utilisant l'IA et les statistiques historiques.\n\n"
        "📊 Fonctionnalités:\n"
        "• Prédiction des scores (mi-temps et match complet)\n"
        "• Statistiques détaillées des équipes\n"
        "• Forme récente des équipes\n"
        "• Prédictions pour les 5 grandes ligues européennes\n\n"
        "🔍 Commandes disponibles:\n"
        "/predire_match - Prédire un match spécifique\n"
        "/ligues - Voir les ligues disponibles\n"
        "/aide - Afficher l'aide\n"
        "/stats - Voir les statistiques globales"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Prédire un match", callback_data='predict')],
        [InlineKeyboardButton("📊 Voir les ligues", callback_data='leagues')],
        [InlineKeyboardButton("❓ Aide", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def show_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les ligues disponibles"""
    leagues = {
        'PL': 'Premier League (Angleterre)',
        'BL1': 'Bundesliga (Allemagne)',
        'SA': 'Serie A (Italie)',
        'PD': 'La Liga (Espagne)',
        'FL1': 'Ligue 1 (France)'
    }
    
    message = "🏆 Ligues disponibles:\n\n"
    for code, name in leagues.items():
        message += f"• {name} ({code})\n"
    
    keyboard = [[InlineKeyboardButton("🔙 Retour", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(message, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /aide - Affiche l'aide"""
    help_text = (
        "📚 Guide d'utilisation du Bot:\n\n"
        "1️⃣ Pour prédire un match:\n"
        "   • Utilisez /predire_match\n"
        "   • Suivez les instructions pour sélectionner la ligue et les équipes\n\n"
        "2️⃣ Pour voir les ligues disponibles:\n"
        "   • Utilisez /ligues\n"
        "   • Choisissez parmi les 5 grandes ligues européennes\n\n"
        "3️⃣ Pour voir les statistiques:\n"
        "   • Utilisez /stats\n"
        "   • Consultez les performances globales\n\n"
        "💡 Conseil: Utilisez les boutons interactifs pour une navigation plus facile!"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Retour", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(help_text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(help_text, reply_markup=reply_markup)

async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques globales"""
    stats = data_manager.get_global_stats()
    
    message = (
        "📊 Statistiques Globales:\n\n"
        f"⚽ Nombre total de matchs analysés: {stats['total_matches']}\n"
        f"🎯 Précision moyenne des prédictions: {stats['accuracy']:.1f}%\n"
        f"🏆 Ligues couvertes: {stats['leagues_covered']}\n"
        f"📈 Données mises à jour: {stats['last_update']}\n\n"
        "💡 Ces statistiques sont basées sur les 3 dernières saisons."
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Retour", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(message, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message, reply_markup=reply_markup)

async def predire_match(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /predire_match - Prédit le score d'un match"""
    keyboard = []
    for code, name in {
        'PL': 'Premier League',
        'BL1': 'Bundesliga',
        'SA': 'Serie A',
        'PD': 'La Liga',
        'FL1': 'Ligue 1'
    }.items():
        keyboard.append([InlineKeyboardButton(name, callback_data=f'league_{code}')])
    
    keyboard.append([InlineKeyboardButton("❌ Annuler", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🏆 Sélectionnez la ligue pour le match:",
        reply_markup=reply_markup
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère les callbacks des boutons inline"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'back':
        keyboard = [
            [InlineKeyboardButton("🎯 Prédire un match", callback_data='predict')],
            [InlineKeyboardButton("📊 Voir les ligues", callback_data='leagues')],
            [InlineKeyboardButton("❓ Aide", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "👋 Bienvenue! Que souhaitez-vous faire?",
            reply_markup=reply_markup
        )
    
    elif query.data == 'predict':
        await predire_match(update, context)
    
    elif query.data == 'leagues':
        await show_leagues(update, context)
    
    elif query.data == 'help':
        await help_command(update, context)
    
    elif query.data == 'cancel':
        await query.edit_message_text(
            "❌ Opération annulée. Utilisez /start pour recommencer.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Retour", callback_data='back')]])
        )
    
    elif query.data.startswith('league_'):
        league = query.data.split('_')[1]
        context.user_data['selected_league'] = league
        await show_teams(update, context, league)

async def show_teams(update: Update, context: ContextTypes.DEFAULT_TYPE, league: str):
    """Affiche les équipes de la ligue sélectionnée"""
    teams = data_manager.get_league_teams(league)
    
    keyboard = []
    for team in teams:
        keyboard.append([InlineKeyboardButton(team['name'], callback_data=f'team_{team["id"]}')])
    
    keyboard.append([InlineKeyboardButton("🔙 Retour", callback_data='predict')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(
        "⚽ Sélectionnez la première équipe:",
        reply_markup=reply_markup
    )

async def handle_team_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère la sélection des équipes"""
    query = update.callback_query
    team_id = query.data.split('_')[1]
    
    if 'team1' not in context.user_data:
        context.user_data['team1'] = team_id
        await show_opponent_teams(update, context)
    else:
        context.user_data['team2'] = team_id
        await make_prediction(update, context)

async def show_opponent_teams(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les équipes adverses possibles"""
    league = context.user_data['selected_league']
    team1_id = context.user_data['team1']
    teams = data_manager.get_league_teams(league)
    
    keyboard = []
    for team in teams:
        if team['id'] != team1_id:
            keyboard.append([InlineKeyboardButton(team['name'], callback_data=f'team_{team["id"]}')])
    
    keyboard.append([InlineKeyboardButton("🔙 Retour", callback_data=f'league_{league}')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(
        "⚽ Sélectionnez la deuxième équipe:",
        reply_markup=reply_markup
    )

async def make_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fait la prédiction du match"""
    query = update.callback_query
    team1_id = context.user_data['team1']
    team2_id = context.user_data['team2']
    
    # Vérification du cache
    cache_key = f"{team1_id}_{team2_id}"
    if cache_key in prediction_cache:
        cache_time, prediction = prediction_cache[cache_key]
        if datetime.now() - cache_time < CACHE_DURATION:
            await send_prediction(update, prediction)
            return
    
    # Récupération des données
    team1_stats = data_manager.get_team_stats(team1_id)
    team2_stats = data_manager.get_team_stats(team2_id)
    
    # Prédiction
    prediction = score_predictor.predict_score(team1_stats, team2_stats)
    
    # Mise en cache
    prediction_cache[cache_key] = (datetime.now(), prediction)
    
    await send_prediction(update, prediction)

async def send_prediction(update: Update, prediction: dict):
    """Envoie la prédiction formatée"""
    ht_home, ht_away = prediction['ht']
    ft_home, ft_away = prediction['ft']
    
    message = (
        "🎯 Prédiction du match:\n\n"
        f"⚽ Mi-temps: {ht_home} - {ht_away}\n"
        f"🏆 Match complet: {ft_home} - {ft_away}\n\n"
        "💡 Cette prédiction est basée sur:\n"
        "• Statistiques historiques\n"
        "• Forme récente des équipes\n"
        "• Modèle d'IA entraîné\n\n"
        "⚠️ Cette prédiction est à titre indicatif uniquement."
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Nouvelle prédiction", callback_data='predict')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(message, reply_markup=reply_markup)

def main():
    """Fonction principale"""
    if not TOKEN:
        logger.error("Token Telegram non trouvé!")
        return

    # Création de l'application
    application = Application.builder().token(TOKEN).build()

    # Ajout des handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predire_match", predire_match))
    application.add_handler(CommandHandler("aide", help_command))
    application.add_handler(CommandHandler("ligues", show_leagues))
    application.add_handler(CommandHandler("stats", show_stats))
    
    # Handler pour les callbacks
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(CallbackQueryHandler(handle_team_selection, pattern='^team_'))

    # Démarrage du bot
    logger.info("Démarrage du bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 