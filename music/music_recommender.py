# music_recommender.py
# VERSÃO FINAL - LÓGICA "BUFFET" CORRIGIDA E SCORE DE SIMILARIDADE

import os
import pandas as pd
import numpy as np
import requests
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, url_for, jsonify, redirect
import traceback
import time
import json

# --- ADICIONE SUAS CREDENCIAIS DO SPOTIFY AQUI ---
SPOTIFY_CLIENT_ID = "x"
SPOTIFY_CLIENT_SECRET = "x"

class SpotifyTokenManager:
    # ... (código do gerenciador de token permanece o mesmo) ...
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry_time = 0
    def get_token(self):
        if self.token and time.time() < self.token_expiry_time: return self.token
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode( )).decode()
        auth_data = {'grant_type': 'client_credentials'}
        try:
            res = requests.post(auth_url, headers={'Authorization': f'Basic {auth_header}'}, data=auth_data)
            res.raise_for_status()
            token_info = res.json()
            self.token = token_info['access_token']
            self.token_expiry_time = time.time() + token_info['expires_in'] - 60
            return self.token
        except requests.exceptions.RequestException as e:
            print(f"ERRO CRÍTICO: Não foi possível obter o token do Spotify. {e}")
            return None

spotify_token_manager = SpotifyTokenManager(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)

class MusicRecommender:
    # ... (inicialização e pré-processamento permanecem os mesmos) ...
    def __init__(self, filepath='data.csv'):
        self.df_music = None; self.feature_matrix = None; self.is_ready = False
        self.text_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.numerical_scaler = MinMaxScaler()
        self.audio_feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness','liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'year']
        try: self._initialize_system(filepath)
        except Exception as e: print(f"\n--- ERRO CRÍTICO NA INICIALIZAÇÃO ---\n{e}"); traceback.print_exc()
    def _initialize_system(self, filepath):
        df = self._load_data(filepath)
        if df is None: return
        self.df_music = self._preprocess_data(df)
        if self.df_music is None: return
        self._create_feature_matrix()
        self.is_ready = True
        print("\n>>> Sistema de recomendação de músicas pronto para uso. <<<\n")
    def _load_data(self, filepath):
        if not os.path.exists(filepath): raise FileNotFoundError(f"Arquivo '{filepath}' não encontrado.")
        df = pd.read_csv(filepath); df.columns = df.columns.str.strip(); return df
    def _preprocess_data(self, df):
        required_cols = ['id', 'name', 'artists']
        if not all(col in df.columns for col in required_cols): raise KeyError(f"Colunas essenciais não encontradas: {[c for c in required_cols if c not in df.columns]}")
        df = df.dropna(subset=required_cols).copy()
        df = df.drop_duplicates(subset=['name', 'artists'])
        df['artists'] = df['artists'].str.replace(r"[\[\]']", "", regex=True)
        df['text_features'] = df['name'] + ' ' + df['artists']
        return df.reset_index(drop=True)
    def _create_feature_matrix(self):
        text_matrix = self.text_vectorizer.fit_transform(self.df_music['text_features'])
        for col in self.audio_feature_cols:
            if col not in self.df_music.columns: raise KeyError(f"Coluna de áudio esperada '{col}' não encontrada.")
            self.df_music[col] = pd.to_numeric(self.df_music[col], errors='coerce').fillna(0)
        numerical_matrix = self.numerical_scaler.fit_transform(self.df_music[self.audio_feature_cols])
        W_TEXT, W_AUDIO = 0.4, 0.6
        self.feature_matrix = np.hstack([text_matrix.toarray() * W_TEXT, numerical_matrix * W_AUDIO])

    def search_tracks(self, query, limit=15):
        if not self.is_ready or query == '': return []
        results = self.df_music[self.df_music['name'].str.contains(query, case=False, na=False) | self.df_music['artists'].str.contains(query, case=False, na=False)]
        return results.sort_values(by='popularity', ascending=False).head(limit).to_dict('records')

    def _create_user_profile(self, selected_track_ids):
        user_indices = self.df_music.index[self.df_music['id'].isin(selected_track_ids)].tolist()
        if not user_indices: raise ValueError("Músicas selecionadas não encontradas.")
        return self.feature_matrix[user_indices].mean(axis=0).reshape(1, -1)

    def get_all_recommendations(self, selected_track_ids):
        user_profile_vector = self._create_user_profile(selected_track_ids)
        
        general_pool_size = 500
        candidate_indices = self.df_music[~self.df_music['id'].isin(selected_track_ids)].index
        similarities = cosine_similarity(user_profile_vector, self.feature_matrix[candidate_indices]).flatten()
        
        recs_pool_df = self.df_music.loc[candidate_indices].copy()
        recs_pool_df['similarity'] = similarities
        recs_pool_df = recs_pool_df.sort_values(by='similarity', ascending=False).head(general_pool_size)

        seen_track_ids = set(selected_track_ids)
        
        def get_unique_recs(df, num_recs, seen_ids):
            recs_df = df[~df['id'].isin(seen_ids)].head(num_recs)
            for rec_id in recs_df['id']: seen_ids.add(rec_id)
            # Adiciona a pontuação de similaridade formatada
            recs_df['similarity_score'] = (recs_df['similarity'] * 100).round(1)
            return recs_df.to_dict('records')

        # Lógica "Buffet"
        main_recs_df = recs_pool_df.head(20)
        gems_recs_df = recs_pool_df[recs_pool_df['popularity'].between(15, 45)].head(10)
        energy_recs_df = recs_pool_df[(recs_pool_df['energy'] > 0.7) & (recs_pool_df['danceability'] > 0.6)].head(10)
        relax_recs_df = recs_pool_df[(recs_pool_df['acousticness'] > 0.7) & (recs_pool_df['energy'] < 0.5)].head(10)

        selected_artists_str = self.df_music[self.df_music['id'].isin(selected_track_ids)]['artists'].str.cat(sep=',')
        main_artists = list(set([artist.strip() for artist in selected_artists_str.split(',') if artist]))
        artist_recs_df = self.df_music[self.df_music['artists'].str.contains('|'.join(main_artists), case=False, na=False) & ~self.df_music['id'].isin(seen_track_ids)].sort_values(by='popularity', ascending=False).head(10)
        # Adiciona similaridade para o dataframe de artistas também
        artist_indices = artist_recs_df.index
        artist_similarities = cosine_similarity(user_profile_vector, self.feature_matrix[artist_indices]).flatten()
        artist_recs_df['similarity'] = artist_similarities

        # Montagem Final
        main_recs = get_unique_recs(main_recs_df, 12, seen_track_ids)
        artist_recs = get_unique_recs(artist_recs_df, 6, seen_track_ids)
        gems_recs = get_unique_recs(gems_recs_df, 6, seen_track_ids)
        energy_recs = get_unique_recs(energy_recs_df, 6, seen_track_ids)
        relax_recs = get_unique_recs(relax_recs_df, 6, seen_track_ids)

        return {'main': main_recs, 'artist': artist_recs, 'gems': gems_recs, 'energy': energy_recs, 'relax': relax_recs}

app = Flask(__name__, template_folder='templates')
recommender = MusicRecommender()

@app.route('/music')
def music_index():
    if not recommender.is_ready: return "Erro: Sistema de recomendação não inicializado.", 500
    return render_template('music_index.html')

@app.route('/music/search', methods=['GET'])
def search_music_api():
    query = request.args.get('q', '')
    if not recommender.is_ready: return jsonify([])
    return jsonify(recommender.search_tracks(query))

@app.route('/music/recommend', methods=['POST'])
def recommend_music_api():
    ids_json_string = request.form.get('track_ids_json')
    if not ids_json_string: return "Erro: Nenhum ID de música recebido.", 400
    try: selected_ids = json.loads(ids_json_string)
    except json.JSONDecodeError: return "Erro: Formato de dados inválido.", 400
    if not selected_ids or len(selected_ids) < 3: return "Erro: Selecione pelo menos 3 músicas.", 400
    
    recommendations = recommender.get_all_recommendations(selected_ids)
    selected_tracks_details = recommender.df_music[recommender.df_music['id'].isin(selected_ids)].to_dict('records')
    
    return render_template('music_results.html', recommendations=recommendations, profile=selected_tracks_details)

@app.route('/music/get-track-details', methods=['POST'])
def get_track_details_api():
    track_ids = request.json.get('track_ids')
    if not track_ids: return jsonify({'error': 'Nenhum ID de música fornecido'}), 400
    token = spotify_token_manager.get_token()
    if not token: return jsonify({'error': 'Falha na autenticação com o Spotify'}), 500
    try:
        headers = {'Authorization': f'Bearer {token}'}
        ids_str = ','.join(list(set(track_ids))[:50])
        url = f'https://api.spotify.com/v1/tracks?ids={ids_str}'
        res = requests.get(url, headers=headers ); res.raise_for_status()
        tracks_data = res.json()['tracks']; results = {}
        for track in tracks_data:
            if track and track['album']['images']: results[track['id']] = track['album']['images'][0]['url']
        return jsonify(results)
    except Exception as e: return jsonify({'error': f"Erro interno ao buscar detalhes: {e}"}), 500

if __name__ == '__main__':
    if recommender.is_ready: app.run(debug=True, port=5002)
    else: print("\nA aplicação não pode iniciar porque o sistema de recomendação falhou ao inicializar.")
