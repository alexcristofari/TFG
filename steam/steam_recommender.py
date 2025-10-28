# steam_recommender.py
# VERSÃO 29.0 - CORREÇÃO DEFINITIVA DO BUG DE RECOMENDAÇÕES DUPLICADAS

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, url_for, redirect, session
import requests
import traceback
import random

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- CONFIGURAÇÕES E CARREGAMENTO DE DADOS ---
API_KEY = "x" 

CURATED_GENRES = [
    'Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Indie',
    'Singleplayer', 'Multiplayer', 'Co-op', 'FPS', 'Shooter', 'Open World',
    'Survival', 'Story Rich', 'Horror', 'Puzzle', 'Platformer', 'Metroidvania',
    'Roguelike', 'Souls-like', 'Building', 'Sandbox', 'Tactical', 'Stealth'
]
CURATED_GENRES.sort()

try:
    print("Carregando dataset 'steam.csv'...")
    df_steam = pd.read_csv('steam.csv')
    print("Dataset carregado com sucesso.")

    print("Pré-processando dados...")
    df_steam = df_steam.dropna(subset=['appid', 'name', 'genres', 'steamspy_tags', 'developer']).copy()
    df_steam['tags_for_vectorizing'] = (df_steam['genres'] + ';' + df_steam['steamspy_tags']).str.replace(';', ' ').str.lower()
    
    df_steam['positive_ratings'] = pd.to_numeric(df_steam['positive_ratings'], errors='coerce').fillna(0)
    df_steam['negative_ratings'] = pd.to_numeric(df_steam['negative_ratings'], errors='coerce').fillna(0)
    df_steam['total_ratings'] = df_steam['positive_ratings'] + df_steam['negative_ratings']
    
    df_steam['rating_score'] = np.where(
        df_steam['total_ratings'] > 100,
        df_steam['positive_ratings'] / df_steam['total_ratings'],
        0
    )
    print("Pré-processamento concluído.")

    print("Criando matriz TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_steam['tags_for_vectorizing'])
    print("Matriz TF-IDF criada com sucesso.")

except FileNotFoundError:
    print("\nERRO CRÍTICO: O arquivo 'steam.csv' não foi encontrado.")
    df_steam = None
except Exception as e:
    print(f"\nERRO CRÍTICO ao carregar ou processar os dados: {e}")
    df_steam = None

# --- FUNÇÕES AUXILIARES DE RECOMENDAÇÃO ---

def _format_recommendation(game_row, score, reason):
    return {
        'name': game_row['name'],
        'developer': game_row['developer'],
        'tags': game_row['steamspy_tags'].split(';') if isinstance(game_row['steamspy_tags'], str) else [],
        'appid': int(game_row['appid']),
        'similarity_score': min(score, 1.0),
        'positive_rating_percent': int(game_row['rating_score'] * 100),
        'recommendation_reason': reason
    }

# ***** INÍCIO DA CORREÇÃO CRÍTICA (LÓGICA DE DE-DUPLICAÇÃO) *****
def get_main_recommendations(selected_appids, owned_appids_set, num_recs=8):
    # Dicionário para armazenar as recomendações únicas, usando appid como chave.
    # Esta é a mudança crucial para evitar duplicatas.
    all_recs = {}
    
    # Pesos para o score combinado
    W_SIM, W_RATING = 0.75, 0.25

    # Itera sobre cada jogo que o usuário selecionou como favorito ("semente")
    for seed_appid in selected_appids:
        try:
            # Encontra o jogo semente no nosso dataset
            seed_index = df_steam.index[df_steam['appid'] == seed_appid].tolist()[0]
            seed_vector = tfidf_matrix[seed_index]
            
            # Calcula a similaridade do cosseno entre a semente e TODOS os outros jogos
            cosine_similarities = cosine_similarity(seed_vector, tfidf_matrix).flatten()
            
            # Cria um score combinado que leva em conta a similaridade e a avaliação do jogo
            combined_score = (W_SIM * cosine_similarities) + (W_RATING * df_steam['rating_score'].values)
            
            # Pega os índices dos 2000 jogos com maior score (para performance)
            top_indices_pool = combined_score.argsort()[:-2000:-1]
            
            # Desses 2000, pega uma amostra aleatória de 800 para introduzir variedade
            sample_size = min(len(top_indices_pool), 800)
            sampled_indices = random.sample(list(top_indices_pool), k=sample_size)
            
            # Ordena a amostra pelo score combinado
            sampled_indices_sorted = sorted(sampled_indices, key=lambda i: combined_score[i], reverse=True)

            recs_found_for_this_seed = 0
            # Itera sobre a amostra ordenada para encontrar recomendações
            for i in sampled_indices_sorted:
                # Limita a 10 recomendações encontradas POR SEMENTE para não enviesar o resultado
                if recs_found_for_this_seed >= 10: 
                    break
                    
                game_row = df_steam.iloc[i]
                rec_appid = int(game_row['appid'])
                
                # Filtra jogos com poucas análises
                if game_row['total_ratings'] < 1000: 
                    continue
                
                # A VERIFICAÇÃO CRÍTICA:
                # Se o jogo não é a própria semente,
                # se o usuário não o possui,
                # E se ele AINDA NÃO FOI ADICIONADO ao nosso dicionário de recomendações...
                if rec_appid != seed_appid and rec_appid not in owned_appids_set and rec_appid not in all_recs:
                    # ...então, e somente então, o adicionamos.
                    score = float(combined_score[i])
                    reason = "Forte alinhamento com seus jogos favoritos."
                    all_recs[rec_appid] = _format_recommendation(game_row.to_dict(), score, reason)
                    recs_found_for_this_seed += 1
        except (IndexError, KeyError):
            # Se o jogo semente não for encontrado no nosso dataset, apenas pulamos para o próximo
            continue
            
    # Finalmente, ordena todas as recomendações únicas encontradas pelo score e retorna o número desejado.
    return sorted(list(all_recs.values()), key=lambda x: x['similarity_score'], reverse=True)[:num_recs]
# ***** FIM DA CORREÇÃO CRÍTICA *****

def get_recommendations_by_genre(selected_genres, owned_appids_set, num_recs_per_genre=4):
    genre_recs_map = {}
    for genre in selected_genres:
        genre_df = df_steam[
            df_steam['genres'].str.contains(genre, case=False, na=False) & 
            (df_steam['total_ratings'] > 5000) &
            ~df_steam['appid'].isin(owned_appids_set)
        ].copy()
        top_games_for_genre = genre_df.sort_values(by='rating_score', ascending=False)
        recs_for_this_genre = []
        for _, game_row in top_games_for_genre.head(num_recs_per_genre).iterrows():
            score = game_row['rating_score']
            reason = f"Um dos jogos mais aclamados do gênero '{genre}'."
            recs_for_this_genre.append(_format_recommendation(game_row.to_dict(), score, reason))
        if recs_for_this_genre:
            genre_recs_map[genre] = recs_for_this_genre
    return genre_recs_map

def get_cult_classics_and_indie_gems(selected_appids, owned_appids_set, seen_appids, num_recs=4):
    all_recs = []
    try:
        selected_indices = df_steam.index[df_steam['appid'].isin(selected_appids)].tolist()
        if not selected_indices: return []
        user_profile_matrix = tfidf_matrix[selected_indices].mean(axis=0)
        user_profile_vector = np.asarray(user_profile_matrix)
        cosine_similarities = cosine_similarity(user_profile_vector, tfidf_matrix).flatten()
        candidate_df = df_steam[
            (df_steam['rating_score'] > 0.90) & 
            (df_steam['total_ratings'].between(2000, 50000)) &
            ~df_steam['appid'].isin(owned_appids_set) &
            ~df_steam['appid'].isin(seen_appids)
        ].copy()
        candidate_indices = candidate_df.index
        if candidate_indices.empty: return []
        candidate_similarities = cosine_similarities[candidate_indices]
        num_to_get = min(len(candidate_similarities), num_recs * 2)
        top_candidates_local_indices = candidate_similarities.argsort()[-num_to_get:][::-1]
        for local_idx in top_candidates_local_indices:
            game_index = candidate_indices[local_idx]
            game_row = df_steam.loc[game_index]
            score = cosine_similarities[game_index]
            reason = "Uma joia indie ou clássico cult com avaliações excelentes."
            all_recs.append(_format_recommendation(game_row.to_dict(), score, reason))
    except Exception as e:
        print(f"Erro em get_cult_classics_and_indie_gems: {e}")
        traceback.print_exc()
        return []
    return all_recs[:num_recs]

# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
def home():
    return redirect(url_for('steam_index'))

@app.route('/steam')
def steam_index():
    if df_steam is None: return "Erro: O dataset não pôde ser carregado.", 500
    return render_template('steam_index.html')

@app.route('/steam/select', methods=['POST'])
def select_games_page():
    steam_id = request.form.get('steam_id')
    if not steam_id: return "SteamID não fornecido.", 400
    session['steam_id'] = steam_id
    try:
        url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={API_KEY}&steamid={steam_id}&format=json&include_appinfo=true"
        response = requests.get(url )
        response.raise_for_status()
        data = response.json().get('response', {})
        owned_games_list = data.get('games', [])
        if not owned_games_list: return "Não foi possível obter os jogos. O perfil pode ser privado.", 404
        owned_games_info = {g['appid']: {'name': g.get('name'), 'playtime_forever': g.get('playtime_forever', 0)} for g in owned_games_list}
        session['owned_games_info'] = owned_games_info
        return render_template('select_games.html', games=owned_games_list, genres=CURATED_GENRES)
    except Exception as e:
        traceback.print_exc()
        return "Ocorreu um erro ao processar sua solicitação.", 500

@app.route('/steam/recommend', methods=['POST'])
def generate_all_recommendations():
    try:
        selected_appids_str = request.form.getlist('selected_games')
        selected_genres = request.form.getlist('selected_genres')
        if not selected_appids_str or len(selected_appids_str) < 3 or not selected_genres:
            return "Seleção inválida.", 400
        selected_appids = [int(appid) for appid in selected_appids_str]
        owned_games_info = session.get('owned_games_info', {})
        owned_appids_set = set(owned_games_info.keys())
        user_profile_details = {
            'selected_games': [owned_games_info[appid] for appid in selected_appids if appid in owned_games_info],
            'selected_genres': selected_genres
        }
        main_recs = get_main_recommendations(selected_appids, owned_appids_set)
        seen_appids = {rec['appid'] for rec in main_recs}
        top_genre_recs_map = get_recommendations_by_genre(selected_genres, owned_appids_set)
        for genre_list in top_genre_recs_map.values():
            for rec in genre_list:
                seen_appids.add(rec['appid'])
        cult_classics_recs = get_cult_classics_and_indie_gems(selected_appids, owned_appids_set, seen_appids)
        recommendations_data = {
            'principais': main_recs,
            'top_generos_map': top_genre_recs_map,
            'cult_classics': cult_classics_recs
        }
        return render_template('results.html', recommendations=recommendations_data, profile=user_profile_details)
    except Exception as e:
        print("\n--- ERRO GRAVE NA ROTA DE RECOMENDAÇÃO ---")
        traceback.print_exc()
        return render_template('results.html', recommendations={}, profile={})

if __name__ == '__main__':
    if df_steam is not None:
        app.run(debug=True)
    else:
        print("A aplicação não pode iniciar porque o dataset não foi carregado.")
