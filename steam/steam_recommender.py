# steam_recommender.py
# VERSÃO FINAL 4.0 - Design Minimalista e Score Normalizado (0-1)

from flask import Flask, render_template, request, session
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# --- Bloco 1 e 2: CONFIGURAÇÃO E CARREGAMENTO (sem alterações) ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
try:
    df_steam = pd.read_csv('steam.csv')
except FileNotFoundError:
    print("ERRO CRÍTICO: 'steam.csv' não foi encontrado.")
    exit()
# (Todo o resto do pré-processamento permanece o mesmo...)
required_cols = ['appid', 'name', 'developer', 'genres', 'steamspy_tags', 'positive_ratings', 'negative_ratings']
for col in required_cols:
    if col not in df_steam.columns:
        print(f"ERRO CRÍTico: A coluna necessária '{col}' não foi encontrada.")
        exit()
df_steam['genres'] = df_steam['genres'].str.replace(';', ' ')
df_steam['steamspy_tags'] = df_steam['steamspy_tags'].str.replace(';', ' ')
text_cols = ['genres', 'steamspy_tags', 'developer']
for col in text_cols:
    df_steam[col] = df_steam[col].fillna('')
df_steam['tags_for_vectorizing'] = df_steam['genres'] + ' ' + df_steam['steamspy_tags'] + ' ' + df_steam['developer']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_steam['tags_for_vectorizing'])
df_steam['rating_score'] = df_steam['positive_ratings'] / (df_steam['positive_ratings'] + df_steam['negative_ratings'] + 1)
STEAM_API_KEY = "x"

# --- Bloco 3: ROTAS E LÓGICA (com alteração na get_recommendations) ---

def get_user_games(steam_id):
    # (Função sem alterações)
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={steam_id}&format=json&include_appinfo=true"
    try:
        response = requests.get(url, timeout=10 )
        response.raise_for_status()
        data = response.json().get('response', {})
        if not data or 'games' not in data: return None
        return [{'appid': game['appid'], 'name': game['name']} for game in data.get('games', [])]
    except requests.exceptions.RequestException:
        return None

@app.route('/steam')
def steam_index():
    return render_template('steam_index.html')

@app.route('/steam/select', methods=['POST'])
def select_games_page():
    # (Função sem alterações)
    steam_id = request.form.get('steam_id')
    if not steam_id: return "Steam ID é obrigatório.", 400
    owned_games = get_user_games(steam_id)
    if owned_games is None: return "Perfil privado ou Steam ID incorreto.", 404
    if not owned_games: return "Biblioteca de jogos vazia.", 404
    session['owned_appids'] = [game['appid'] for game in owned_games]
    return render_template('select_games.html', games=owned_games)

def generate_recommendation_reason(score):
    if score >= 0.9: return "Uma combinação perfeita com seu perfil de jogador!"
    elif score >= 0.7: return "Este jogo combina muito bem com suas preferências."
    else: return "Este jogo tem várias características que você pode gostar."

@app.route('/steam/recommend', methods=['POST'])
def get_recommendations():
    # (Lógica inicial sem alterações)
    selected_appids = [int(appid) for appid in request.form.getlist('selected_games')]
    if len(selected_appids) < 3: return "Selecione pelo menos 3 jogos.", 400
    selected_games_df = df_steam[df_steam['appid'].isin(selected_appids)]
    if selected_games_df.empty: return "Jogos selecionados não encontrados em nosso banco de dados.", 404
    selected_indices = selected_games_df.index.tolist()
    user_profile_vector = np.mean(tfidf_matrix[selected_indices], axis=0)
    user_profile_vector = np.asarray(user_profile_vector)
    cosine_similarities = cosine_similarity(user_profile_vector, tfidf_matrix)
    combined_score = cosine_similarities[0] * (df_steam['rating_score'].values + 0.1)
    similar_indices = combined_score.argsort()[:-100:-1]

    # --- INÍCIO DA ALTERAÇÃO ---
    recommendations_with_scores = []
    owned_appids = set(session.get('owned_appids', []))

    for i in similar_indices:
        appid = int(df_steam.iloc[i]['appid'])
        if appid not in owned_appids and appid not in selected_appids:
            recommendations_with_scores.append({
                'details': df_steam.iloc[i].to_dict(),
                'score': float(combined_score[i])
            })
        if len(recommendations_with_scores) >= 15:
            break
    
    # Normalização do Score para a escala 0-1
    max_score = max(rec['score'] for rec in recommendations_with_scores) if recommendations_with_scores else 1
    
    recommendations = []
    for rec in recommendations_with_scores:
        normalized_score = rec['score'] / max_score
        game_row = rec['details']
        recommendations.append({
            'name': game_row['name'],
            'developer': game_row['developer'],
            'tags': game_row['steamspy_tags'].split(';'),
            'appid': int(game_row['appid']),
            'similarity_score': normalized_score,
            'recommendation_reason': generate_recommendation_reason(normalized_score)
        })
    # --- FIM DA ALTERAÇÃO ---
            
    return render_template('results.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
