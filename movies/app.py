# movies/app.py (v10.2 - Estrutura Padronizada)
# VERSÃO 10.2 - Ajusta caminhos de cache e padroniza nome do arquivo.

import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for
import traceback, json, numpy as np, pickle, os # Adicionado 'os'
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from thefuzz import fuzz
from datetime import datetime

class MovieRecommender:
    def __init__(self):
        self.df_movies = None; self.tfidf_matrix = None; self.is_ready = False
        self.QUANTILE_95_POPULARITY = 0; self.GENRES_LIST = []
        try: self._initialize_from_cache()
        except Exception as e: print(f"\n--- ERRO AO CARREGAR CACHE ---\nVerifique se rodou 'build_movie_cache.py'.\n{e}")

    def _initialize_from_cache(self):
        print("Carregando cache de filmes...")
        
        # --- ALTERAÇÃO PRINCIPAL AQUI ---
        CACHE_DIR = 'cache'
        self.df_movies = pd.read_parquet(os.path.join(CACHE_DIR, 'movies_processed.parquet'))
        with open(os.path.join(CACHE_DIR, 'movie_tfidf_matrix.pkl'), 'rb') as f: self.tfidf_matrix = pickle.load(f)
        with open(os.path.join(CACHE_DIR, 'genres.json'), 'r', encoding='utf-8') as f: self.GENRES_LIST = json.load(f)
        # --- FIM DA ALTERAÇÃO ---

        self.df_movies['release_date_dt'] = pd.to_datetime(self.df_movies['release_date'], errors='coerce')
        self.QUANTILE_95_POPULARITY = self.df_movies['popularity'].quantile(0.95)
        self.is_ready = True
        print(f"\n>>> Sistema de filmes pronto. {len(self.df_movies)} filmes e {len(self.GENRES_LIST)} gêneros carregados. <<<\n")

    def search_movies(self, query, limit=20):
        if not self.is_ready or not query: return []
        query = query.lower()
        results_df = self.df_movies[self.df_movies['search_features'].str.contains(query, na=False)]
        sorted_results = results_df.sort_values(by='popularity', ascending=False).head(limit)
        return sorted_results.assign(poster_url='https://image.tmdb.org/t/p/w500' + sorted_results['poster_path']  ).to_dict('records')

    def discover_movies(self):
        if not self.is_ready: return {}
        NUM_DISCOVER_MOVIES = 22
        two_years_ago = datetime.now().year - 2
        recent_movies = self.df_movies[self.df_movies['release_date_dt'].dt.year >= two_years_ago]
        popular_releases = recent_movies.sort_values(by='popularity', ascending=False).head(NUM_DISCOVER_MOVIES)
        critically_acclaimed = self.df_movies[self.df_movies['vote_average'] > 8.0].sort_values(by='popularity', ascending=False).head(NUM_DISCOVER_MOVIES)
        return {
            "popular_releases": popular_releases.assign(poster_url='https://image.tmdb.org/t/p/w500' + popular_releases['poster_path']  ).to_dict('records'),
            "critically_acclaimed": critically_acclaimed.assign(poster_url='https://image.tmdb.org/t/p/w500' + critically_acclaimed['poster_path']  ).to_dict('records')
        }

    def _calculate_hybrid_score(self, df):
        SIMILARITY_WEIGHT = 0.70; QUALITY_WEIGHT = 0.30; SIMILARITY_CEILING = 0.35
        sim_score = (df['similarity'] / SIMILARITY_CEILING * 95).clip(upper=99)
        quality_score = df['vote_average'] * 10
        df['hybrid_score'] = (sim_score * SIMILARITY_WEIGHT) + (quality_score * QUALITY_WEIGHT)
        return df

    def _finalize_recommendations(self, df, num_recs, score_column='hybrid_score'):
        if df.empty or score_column not in df.columns: return []
        df_sorted = df.sort_values(by=score_column, ascending=False).copy()
        final_recs_list = []
        for _, movie in df_sorted.head(num_recs * 3).iterrows():
            is_redundant = False
            for rec in final_recs_list:
                if fuzz.token_sort_ratio(movie['title'], rec['title']) > 90: is_redundant = True; break
            if not is_redundant: final_recs_list.append(movie.to_dict())
            if len(final_recs_list) >= num_recs: break
        if not final_recs_list: return []
        final_df = pd.DataFrame(final_recs_list)
        if len(final_df) >= 1: final_df.loc[final_df.index[0], 'final_score'] = np.random.uniform(98, 99)
        if len(final_df) >= 2: final_df.loc[final_df.index[1], 'final_score'] = np.random.uniform(96, 97)
        if len(final_df) >= 3: final_df.loc[final_df.index[2], 'final_score'] = np.random.uniform(95, 96)
        if len(final_df) > 3:
            rest_df = final_df.iloc[3:].copy()
            min_score = rest_df[score_column].min(); max_score = rest_df[score_column].max()
            if max_score > min_score:
                score_norm = (rest_df[score_column] - min_score) / (max_score - min_score)
                final_df.loc[rest_df.index, 'final_score'] = 70 + (score_norm ** 1.5) * (94 - 70)
            else: final_df.loc[rest_df.index, 'final_score'] = 82
        final_df['similarity_score'] = final_df['final_score'].round(0).astype(int)
        final_df['poster_url'] = 'https://image.tmdb.org/t/p/w500' + final_df['poster_path']
        def get_genre_names(genres_json  ):
            try: return [g.strip() for g in str(genres_json).split(',')]
            except: return []
        final_df['genres_list'] = final_df['genres'].apply(get_genre_names)
        return final_df.to_dict('records')

    def recommend_movie_categories(self, selected_movie_ids, selected_genre):
        if not self.is_ready or not selected_movie_ids: return {}
        selected_movies = self.df_movies[self.df_movies['id'].isin(selected_movie_ids)]
        if selected_movies.empty: return {}
        user_profile = self.tfidf_matrix[selected_movies.index].mean(axis=0)
        exclude_ids = set(selected_movie_ids)
        all_recs_df = self.df_movies[~self.df_movies['id'].isin(exclude_ids)].copy()
        user_profile_array = np.asarray(user_profile)
        cosine_similarities = cosine_similarity(user_profile_array, self.tfidf_matrix[all_recs_df.index]).flatten()
        all_recs_df['similarity'] = cosine_similarities
        all_recs_df = self._calculate_hybrid_score(all_recs_df)
        main_recs = self._finalize_recommendations(all_recs_df, 12)
        exclude_ids.update([rec['id'] for rec in main_recs])
        genre_favorites = self.recommend_by_genre(selected_genre, exclude_ids)
        exclude_ids.update([rec['id'] for rec in genre_favorites])
        blockbusters = self._finalize_recommendations(all_recs_df[~all_recs_df['id'].isin(exclude_ids) & (all_recs_df['popularity'] > self.QUANTILE_95_POPULARITY)], 6)
        exclude_ids.update([rec['id'] for rec in blockbusters])
        cult_classics = self._finalize_recommendations(all_recs_df[~all_recs_df['id'].isin(exclude_ids) & (self.df_movies['release_date_dt'].dt.year < 2005) & (all_recs_df['vote_average'] > 7.0)], 6)
        exclude_ids.update([rec['id'] for rec in cult_classics])
        hidden_gems = self._finalize_recommendations(all_recs_df[~all_recs_df['id'].isin(exclude_ids) & (all_recs_df['vote_average'] > 7.5) & (all_recs_df['popularity'] < self.df_movies['popularity'].quantile(0.7)) & (all_recs_df['popularity'] > self.df_movies['popularity'].quantile(0.3))], 6)
        return {"main": main_recs, "blockbusters": blockbusters, "genre_favorites": genre_favorites, "cult_classics": cult_classics, "hidden_gems": hidden_gems}

    def recommend_by_genre(self, genre_name, exclude_ids):
        if not genre_name: return []
        genre_df = self.df_movies[self.df_movies['genres'].str.contains(genre_name, na=False, case=False)].copy()
        genre_df = genre_df[~genre_df['id'].isin(exclude_ids)]
        QUALITY_WEIGHT = 0.80; POPULARITY_WEIGHT = 0.20
        pop_max = genre_df['popularity'].max()
        pop_score = (genre_df['popularity'] / pop_max * 100) if pop_max > 0 else 0
        quality_score = genre_df['vote_average'] * 10
        genre_df['hybrid_score'] = (quality_score * QUALITY_WEIGHT) + (pop_score * POPULARITY_WEIGHT)
        return self._finalize_recommendations(genre_df, 6, score_column='hybrid_score')

    def analyze_user_profile(self, selected_movie_ids):
        selected_movies = self.df_movies[self.df_movies['id'].isin(selected_movie_ids)]
        if selected_movies.empty: return None, [], []
        all_genres = []
        for genres_str in selected_movies['genres']:
            all_genres.extend([g.strip() for g in str(genres_str).split(',') if g.strip()])
        if not all_genres: return None, [], selected_movies.to_dict('records')
        genre_counts = Counter(all_genres)
        favorite_genre = genre_counts.most_common(1)[0][0]
        unique_genres = sorted(list(genre_counts.keys()))
        return favorite_genre, unique_genres, selected_movies.to_dict('records')

# --- FLASK APP e Rotas ---
app = Flask(__name__, template_folder='templates')
movie_recommender = MovieRecommender()
@app.route('/')
def root(): return redirect(url_for('movies_index'))
@app.route('/movies')
def movies_index():
    return render_template('movies_index.html', genres=movie_recommender.GENRES_LIST)
@app.route('/movies/discover')
def discover_movies_api():
    return jsonify(movie_recommender.discover_movies())
@app.route('/movies/search', methods=['GET'])
def search_movies_api(): return jsonify(movie_recommender.search_movies(request.args.get('q', '')))
@app.route('/movies/recommend', methods=['POST'])
def recommend_movies_api():
    ids_json_string = request.form.get('movie_ids_json')
    selected_genre = request.form.get('genre_select')
    try: selected_ids = json.loads(ids_json_string)
    except: return "Erro nos dados", 400
    favorite_genre, unique_genres, selected_movies_details = movie_recommender.analyze_user_profile(selected_ids)
    recommendations = movie_recommender.recommend_movie_categories(selected_ids, selected_genre)
    profile_data = {"movies": selected_movies_details, "favorite_genre": favorite_genre, "unique_genres": unique_genres}
    return render_template('movies_results.html', recommendations=recommendations, profile=profile_data, selected_genre=selected_genre)

if __name__ == '__main__':
    if movie_recommender.is_ready: app.run(debug=True, port=5001)
    else: print("\nAplicação não pode iniciar. Rode 'build_movie_cache.py' primeiro.")
