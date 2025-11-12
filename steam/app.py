# steam/app.py (v4.2 - Padrão Ouro Adaptado)
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process
from flask import Flask, request, jsonify, render_template, url_for
import json
import traceback
from collections import defaultdict
import os

# --- Configurações de Caminho ---
CACHE_DIR = 'cache'

# --- FUNÇÃO DE LIMPEZA ---
def sanitize_for_json(data):
    if isinstance(data, (list, tuple)): return [sanitize_for_json(item) for item in data]
    if isinstance(data, dict): return {key: sanitize_for_json(value) for key, value in data.items()}
    if isinstance(data, np.ndarray): return data.tolist()
    if isinstance(data, (np.int64, np.int32, np.int16)): return int(data)
    if isinstance(data, (np.float64, np.float32)): return float(data)
    if pd.isna(data): return None
    return data

# --- Classe GameRecommender (Padrão Ouro Adaptado) ---
class GameRecommender:
    def __init__(self):
        self.is_ready = False
        try:
            print("Carregando cache de jogos (v4.2 - Padrão Ouro Adaptado)...")
            self.df = pd.read_parquet(os.path.join(CACHE_DIR, 'games_processed_df.parquet'))
            
            with open(os.path.join(CACHE_DIR, 'genres_matrix.pkl'), 'rb') as f: self.genres_matrix = pickle.load(f)
            with open(os.path.join(CACHE_DIR, 'categories_matrix.pkl'), 'rb') as f: self.categories_matrix = pickle.load(f)
            with open(os.path.join(CACHE_DIR, 'description_matrix.pkl'), 'rb') as f: self.description_matrix = pickle.load(f)
            with open(os.path.join(CACHE_DIR, 'developers_matrix.pkl'), 'rb') as f: self.developers_matrix = pickle.load(f)
            
            with open(os.path.join(CACHE_DIR, 'games_genres.json'), 'r', encoding='utf-8') as f: self.genres = json.load(f)
            
            weights = {'genres': 4.0, 'categories': 3.0, 'description': 1.5, 'developers': 0.5}
            self.feature_matrix = (
                self.genres_matrix * weights['genres'] + self.categories_matrix * weights['categories'] +
                self.description_matrix * weights['description'] + self.developers_matrix * weights['developers']
            )
            
            # Adiciona a coluna de ano para o filtro de "Clássicos"
            self.df['release_year'] = pd.to_datetime(self.df['release_date'], errors='coerce').dt.year
            
            self.is_ready = True
            print(f">>> Sistema de jogos pronto. {len(self.df)} jogos e {len(self.genres)} gêneros carregados. <<<")
        except FileNotFoundError as e:
            print(f"\n--- ERRO CRÍTICO: Arquivo de cache não encontrado: {e}. ---")
        except Exception:
            print("\n--- ERRO CRÍTICO DURANTE A INICIALIZAÇÃO ---")
            traceback.print_exc()

    def get_df_as_records(self, df_to_convert):
        if df_to_convert is None or df_to_convert.empty: return []
        records = df_to_convert.to_dict('records')
        return sanitize_for_json(records)

    def search_games(self, query, limit=30):
        if not self.is_ready or not query: return []
        choices = self.df['name']
        results = process.extractBests(query, choices, score_cutoff=60, limit=limit)
        if not results: return []
        result_indices = [r[2] for r in results]
        return self.get_df_as_records(self.df.iloc[result_indices])

    def discover_games(self):
        if not self.is_ready: return {}, {}
        iconic_appids = [570, 730, 271590, 1091500, 292030, 1245620, 620, 413150]
        iconic_games = self.df[self.df['appid'].isin(iconic_appids)]
        explore_df = self.df[
            (self.df['quality'] > 0.92) &
            (~self.df['genres'].apply(lambda g: isinstance(g, list) and any(main_g in g for main_g in ['Ação', 'Aventura', 'RPG', 'Estratégia'])))
        ].sample(n=15, random_state=42)
        return self.get_df_as_records(iconic_games), self.get_df_as_records(explore_df)

    def get_recommendations(self, selected_game_ids):
        if not self.is_ready or not selected_game_ids: return pd.DataFrame()
        
        selected_indices = self.df.index[self.df['appid'].isin(selected_game_ids)].tolist()
        if not selected_indices: return pd.DataFrame()
        
        # 1. Calcular o vetor de perfil do usuário (média dos vetores dos jogos selecionados)
        user_profile_vector = self.feature_matrix[selected_indices].mean(axis=0)
        
        # 2. Calcular a similaridade de cosseno entre o perfil do usuário e todos os jogos
        similarities = cosine_similarity(user_profile_vector, self.feature_matrix).flatten()
        
        recs_df = self.df.copy()
        recs_df['similarity'] = similarities
        recs_df = recs_df[~recs_df['appid'].isin(selected_game_ids)]
        
        # 3. Calcular o score híbrido (70% Similaridade + 30% Qualidade)
        recs_df['hybrid_score'] = (recs_df['similarity'] * 0.7) + (recs_df['quality'] * 0.3)
        
        # 4. Aplicar penalidade de desenvolvedor
        developer_penalty_factor = 0.85
        developer_counts = defaultdict(int)
        penalized_scores = []
        final_df = recs_df.sort_values('hybrid_score', ascending=False)
        
        for _, row in final_df.iterrows():
            developers_list = row['developers']
            developer = developers_list[0] if isinstance(developers_list, list) and developers_list else 'N/A'
            penalty = developer_penalty_factor ** developer_counts[developer]
            penalized_scores.append(row['hybrid_score'] * penalty)
            developer_counts[developer] += 1
        
        final_df['penalized_score'] = penalized_scores
        
        # 5. Normalizar o score final para 0-100% e retornar
        max_score = final_df['penalized_score'].max()
        if max_score > 0:
            final_df['penalized_score'] = final_df['penalized_score'] / max_score
        
        return final_df.sort_values('penalized_score', ascending=False)

# --- Rotas Flask (Padrão Ouro Adaptado) ---
app = Flask(__name__, template_folder='templates')
recommender = GameRecommender()

@app.route('/games')
def games_index():
    if not recommender.is_ready: return "Erro: Cache de jogos não construído.", 500
    return render_template('games_index.html', 
                           genres=recommender.genres, 
                           search_url=url_for('search_games_api'), 
                           discover_url=url_for('discover_games_api'),
                           recommend_url=url_for('recommend_games_api'))

@app.route('/games/search')
def search_games_api():
    if not recommender.is_ready: return jsonify({"error": "Sistema não pronto"}), 500
    query = request.args.get('q', '')
    results_json = recommender.search_games(query)
    return jsonify(results_json)

@app.route('/games/discover')
def discover_games_api():
    if not recommender.is_ready: return jsonify({"error": "Sistema não pronto"}), 500
    iconic_games_json, explore_games_json = recommender.discover_games()
    return jsonify({
        "iconic_games": iconic_games_json, 
        "explore_games": explore_games_json
    })

@app.route('/games/recommend', methods=['POST'])
def recommend_games_api():
    if not recommender.is_ready: return "Sistema não pronto", 500
    
    form_data = request.form
    game_ids_json = form_data.get('game_ids_json')
    selected_genre_to_explore = form_data.get('genre') or None
    
    try: selected_ids = [int(gid) for gid in json.loads(game_ids_json)]
    except: return "Formato de dados inválido", 400
    if not selected_ids or len(selected_ids) < 3: return "Selecione pelo menos 3 jogos", 400

    recs_df = recommender.get_recommendations(selected_ids)
    
    profile_df = recommender.df[recommender.df['appid'].isin(selected_ids)]
    profile_all_genres = sorted(list(set(g for _, row in profile_df.iterrows() for g in row['genres'] if g)))
    dominant_genre_series = pd.Series([g for _, row in profile_df.iterrows() for g in row['genres'] if g]).mode()
    dominant_genre = dominant_genre_series[0] if not dominant_genre_series.empty else None

    used_ids = set(selected_ids)
    def get_unique_recs(df, num_recs):
        nonlocal used_ids
        if df is None or df.empty: return []
        unique_df = df[~df['appid'].isin(used_ids)].head(num_recs)
        if unique_df.empty: return []
        
        used_ids.update(unique_df['appid'].tolist())
        return recommender.get_df_as_records(unique_df)

    recommendations = {}
    
    recommendations["Recomendações Principais"] = get_unique_recs(recs_df, 12)

    if selected_genre_to_explore:
        explore_recs_df = recs_df[recs_df['genres'].apply(lambda g_list: isinstance(g_list, list) and selected_genre_to_explore in g_list)]
        recommendations[f"Explorando {selected_genre_to_explore}"] = get_unique_recs(explore_recs_df, 6)

    if dominant_genre and dominant_genre != selected_genre_to_explore:
        dominant_genre_recs_df = recs_df[recs_df['genres'].apply(lambda g_list: isinstance(g_list, list) and dominant_genre in g_list)]
        recommendations[f"Com Base no seu Gosto em {dominant_genre}"] = get_unique_recs(dominant_genre_recs_df, 6)

    classic_games_df = recs_df[recs_df['release_year'] < 2018]
    recommendations["Clássicos do seu Estilo"] = get_unique_recs(classic_games_df, 6)

    for category in recommendations:
        for rec in recommendations[category]:
            rec['similarity_score'] = f"{rec.get('penalized_score', 0) * 100:.1f}"

    profile_data = {
        "games": recommender.get_df_as_records(profile_df),
        "dominant_genre": dominant_genre,
        "all_genres": profile_all_genres
    }

    return render_template('games_results.html', 
                           recommendations=recommendations, 
                           profile=profile_data, 
                           selected_genre=selected_genre_to_explore,
                           home_url=url_for('games_index'))

if __name__ == '__main__':
    if recommender.is_ready:
        app.run(debug=True, port=5000)
    else:
        print("\nA aplicação não pode iniciar. Rode o 'build_games_cache.py' primeiro.")