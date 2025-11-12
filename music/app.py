# music/app.py (v13.3 - Caminho de Cache de Token Corrigido)
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, jsonify
import json
import os
import traceback
import requests
import base64
import time
from collections import defaultdict

# --- SPOTIFY API CREDENTIALS ---
SPOTIFY_CLIENT_ID = "6b5cdcb34b384ce795186aae26220918"
SPOTIFY_CLIENT_SECRET = "bd24ed38cfbd46b0ba1243132a1a7c38"

# --- Gerenciador de Token do Spotify ---
class SpotifyTokenManager:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        # --- ALTERAÇÃO AQUI: Caminho do arquivo de cache atualizado ---
        self.token_cache_file = os.path.join('cache', '.spotify_token_cache')
        # --- FIM DA ALTERAÇÃO ---
        self.token_info = self._load_token_from_cache()

    def _load_token_from_cache(self):
        try:
            # Garante que o diretório de cache exista antes de tentar ler
            os.makedirs(os.path.dirname(self.token_cache_file), exist_ok=True)
            with open(self.token_cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_token_to_cache(self, token_info):
        token_info['expires_at'] = time.time() + token_info['expires_in']
        with open(self.token_cache_file, 'w') as f:
            json.dump(token_info, f)
        self.token_info = token_info

    def get_token(self):
        if self.token_info and time.time() < self.token_info.get('expires_at', 0):
            return self.token_info['access_token']
        
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode( )).decode()
        auth_data = {'grant_type': 'client_credentials'}
        try:
            res = requests.post(auth_url, headers={'Authorization': f'Basic {auth_header}'}, data=auth_data, timeout=10)
            res.raise_for_status()
            token_info = res.json()
            self._save_token_to_cache(token_info)
            return token_info['access_token']
        except requests.exceptions.RequestException as e:
            print(f"ERRO CRÍTICO: Não foi possível obter o token do Spotify. {e}")
            return None

spotify_token_manager = SpotifyTokenManager(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)

# --- Classe Principal do Recomendador ---
class MusicRecommender:
    def __init__(self):
        self.df_music = None; self.feature_matrix = None; self.all_genres = []
        self.is_ready = False
        try:
            print("Carregando cache de músicas...")
            CACHE_DIR = 'cache'
            self.df_music = pd.read_parquet(os.path.join(CACHE_DIR, 'music_processed.parquet'))
            with open(os.path.join(CACHE_DIR, 'music_feature_matrix.pkl'), 'rb') as f: self.feature_matrix = pickle.load(f)
            with open(os.path.join(CACHE_DIR, 'music_genres.json'), 'r', encoding='utf-8') as f: self.all_genres = json.load(f)
            self.is_ready = True
            print(f">>> Sistema de músicas pronto. {len(self.df_music)} faixas carregadas. <<<")
        except FileNotFoundError:
            print("\n--- ERRO CRÍTICO: Arquivo de cache não encontrado. ---\nExecute o script 'music/build_music_cache.py' primeiro.")
        except Exception as e:
            print(f"\n--- ERRO CRÍTICO NA INICIALIZAÇÃO ---\n{e}"); traceback.print_exc()

    def search_tracks(self, query, limit=30):
        if not self.is_ready or not query: return []
        search_series = self.df_music['name'] + " " + self.df_music['artists']
        results_df = self.df_music[search_series.str.contains(query, case=False, na=False)]
        return results_df.head(limit).to_dict('records')

    def get_recommendations(self, selected_track_ids, top_n_per_item=20):
        if not self.is_ready or not selected_track_ids: return pd.DataFrame()
        
        all_recs_dfs = []
        selected_indices = self.df_music.index[self.df_music['id'].isin(selected_track_ids)].tolist()
        if not selected_indices: return pd.DataFrame()

        for track_index in selected_indices:
            track_vector = self.feature_matrix[track_index]
            similarities = cosine_similarity(track_vector, self.feature_matrix).flatten()
            
            recs_for_item = self.df_music.copy()
            recs_for_item['similarity'] = similarities
            recs_for_item = recs_for_item[~recs_for_item['id'].isin(selected_track_ids)]
            
            top_recs = recs_for_item.nlargest(top_n_per_item, 'similarity')
            all_recs_dfs.append(top_recs)

        if not all_recs_dfs: return pd.DataFrame()

        combined_recs_df = pd.concat(all_recs_dfs).reset_index(drop=True)
        combined_recs_df = combined_recs_df.sort_values('similarity', ascending=False).drop_duplicates(subset=['id'])
        
        combined_recs_df['similarity'] = combined_recs_df['similarity'] ** 4

        artist_penalty_factor = 0.85
        artist_counts = defaultdict(int)
        penalized_scores = []
        final_df = combined_recs_df.sort_values('similarity', ascending=False)

        for _, row in final_df.iterrows():
            artist = row['artists']
            penalty = artist_penalty_factor ** artist_counts[artist]
            penalized_scores.append(row['similarity'] * penalty)
            artist_counts[artist] += 1
        
        final_df['penalized_score'] = penalized_scores
        return final_df.sort_values('penalized_score', ascending=False)

    def discover_tracks(self):
        if not self.is_ready: return {}, {}
        iconic_artists = ['Arctic Monkeys', 'Billie Eilish', 'The Weeknd', 'Daft Punk', 'Queen', 'Kendrick Lamar', 'Tame Impala', 'Radiohead', 'Red Hot Chili Peppers', 'Foo Fighters']
        iconic_tracks = self.df_music[self.df_music['artists'].isin(iconic_artists)]
        iconic_tracks = iconic_tracks.loc[iconic_tracks.groupby('artists')['popularity'].idxmax()]
        explore_df = self.df_music[(self.df_music['popularity'].between(60, 80)) & (~self.df_music['genres'].isin(['pop', 'dance', 'rock', 'hip-hop', 'latin']))].sample(n=15, random_state=42)
        return iconic_tracks.to_dict('records'), explore_df.to_dict('records')

app = Flask(__name__, template_folder='templates')
recommender = MusicRecommender()

@app.route('/music')
def music_index():
    if not recommender.is_ready: return "Erro: Sistema de recomendação não inicializado.", 500
    return render_template('music_index.html', genres=recommender.all_genres)

@app.route('/music/discover')
def discover_tracks_api():
    if not recommender.is_ready: return jsonify({"error": "Sistema não pronto"}), 500
    iconic_tracks_json, explore_tracks_json = recommender.discover_tracks()
    return jsonify({"iconic_tracks": iconic_tracks_json, "explore_tracks": explore_tracks_json})

@app.route('/music/search', methods=['GET'])
def search_music_api():
    query = request.args.get('q', '')
    if not recommender.is_ready: return jsonify([])
    return jsonify(recommender.search_tracks(query))

@app.route('/music/recommend', methods=['POST'])
def recommend_music_api():
    if not recommender.is_ready: return "Sistema não pronto", 500
    form_data = request.form; ids_json_string = form_data.get('track_ids_json')
    selected_genre_to_explore = form_data.get('genre') or None
    try: selected_ids = json.loads(ids_json_string)
    except: return "Formato de dados inválido", 400
    if not selected_ids or len(selected_ids) < 3: return "Selecione pelo menos 3 músicas", 400
    recs_df = recommender.get_recommendations(selected_ids)
    profile_df = recommender.df_music[recommender.df_music['id'].isin(selected_ids)]
    profile_all_genres = sorted(profile_df['genres'].unique().tolist())
    dominant_genre = profile_df['genres'].mode()[0] if not profile_df['genres'].mode().empty else None
    used_ids = set(selected_ids)
    def get_unique_recs(df, num_recs):
        nonlocal used_ids
        if df.empty: return []
        unique_df = df[~df['id'].isin(used_ids)].head(num_recs)
        used_ids.update(unique_df['id'].tolist())
        return unique_df.to_dict('records')
    recommendations = {}
    recommendations["Recomendações Principais"] = get_unique_recs(recs_df, 12)
    if selected_genre_to_explore:
        explore_recs_df = recs_df[recs_df['genres'] == selected_genre_to_explore]
        recommendations[f"Explorando {selected_genre_to_explore.title()}"] = get_unique_recs(explore_recs_df, 6)
    if dominant_genre and dominant_genre != selected_genre_to_explore:
        dominant_genre_recs_df = recs_df[recs_df['genres'] == dominant_genre]
        recommendations[f"Com Base no seu Gosto em {dominant_genre.title()}"] = get_unique_recs(dominant_genre_recs_df, 6)
    hidden_gems_df = recs_df[recs_df['popularity'] < 50]
    recommendations["Joias Escondidas para Você"] = get_unique_recs(hidden_gems_df, 6)
    max_score = recs_df['penalized_score'].max() if not recs_df.empty else 1.0
    for category in recommendations:
        for rec in recommendations[category]:
            score = rec.get('penalized_score', 0)
            normalized_score = (score / max_score) * 100 if max_score > 0 else 0
            rec['similarity_score'] = f"{min(normalized_score, 100.0):.1f}"
    profile_data = {"tracks": profile_df[['name', 'artists']].to_dict('records'), "dominant_genre": dominant_genre, "all_genres": profile_all_genres}
    return render_template('music_results.html', recommendations=recommendations, profile=profile_data, selected_genre=selected_genre_to_explore)

@app.route('/music/get-track-details', methods=['POST'])
def get_track_details_api():
    track_ids = request.json.get('track_ids')
    if not track_ids: return jsonify({}), 400
    token = spotify_token_manager.get_token()
    if not token: return jsonify({'error': 'Falha na autenticação com o Spotify'}), 500
    headers = {'Authorization': f'Bearer {token}'}
    ids_str = ','.join(list(set(track_ids))[:50])
    url = f'https://api.spotify.com/v1/tracks?ids={ids_str}'
    try:
        res = requests.get(url, headers=headers, timeout=10 )
        res.raise_for_status(); tracks_data = res.json().get('tracks', [])
        results = {}
        for track in tracks_data:
            if track:
                results[track['id']] = {
                    'image_url': track['album']['images'][0]['url'] if track.get('album') and track['album'].get('images') else None,
                    'preview_url': track.get('preview_url')
                }
        return jsonify(results)
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar detalhes no Spotify: {e}"); return jsonify({'error': str(e)}), 502

if __name__ == '__main__':
    if recommender.is_ready: app.run(debug=True, port=5002)
    else: print("\nA aplicação não pode iniciar. Verifique os erros de inicialização.")
