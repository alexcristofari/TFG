import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional, Generator
import time
import datetime
import os
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import traceback
import random
import json

# Fila de progresso para comunicação entre threads
progress_queue = []

def send_progress(message: str):
    """Adiciona uma mensagem de progresso à fila global e imprime no terminal."""
    print(message)
    progress_queue.append(message)

class MovieRecommendationSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.scaler = MinMaxScaler( )
        self.user_profile: Optional[Dict] = None
        self.user_features_vector: Optional[np.ndarray] = None
        self.all_genres: List[str] = []
        self.genre_id_map: Dict[str, int] = {}
        self.movie_details_cache: Dict[int, Dict] = {}
        self.liked_movie_ids_history: List[int] = []
        self.feature_weights = self._get_feature_weights()
        self.get_genres()

    def _get_feature_weights(self) -> np.ndarray:
        num_genres = 19
        genre_weights = [2.0] * num_genres
        other_weights = [1.0, 0.5, 0.7, 0.2, 0.3, 0.6]
        return np.array(genre_weights + other_weights)

    def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{self.base_url}/{endpoint}"
        default_params = {"api_key": self.api_key, "language": "pt-BR"}
        if params:
            default_params.update(params)
        
        try:
            response = requests.get(url, params=default_params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            send_progress(f"Erro na requisição à API do TMDb: {e}")
            return None

    def get_genres(self) -> List[Dict]:
        send_progress("[SETUP] Coletando lista de gêneros do TMDb...")
        data = self._make_api_request("genre/movie/list")
        if data and "genres" in data:
            self.all_genres = [g["name"] for g in data["genres"]]
            self.genre_id_map = {g["name"]: g["id"] for g in data["genres"]}
            if len(self.all_genres) != 19:
                self.feature_weights = self._get_feature_weights()
            send_progress(f"✓ {len(self.all_genres)} gêneros carregados.")
            return data["genres"]
        return []

    def get_movie_details(self, movie_id: int) -> Dict:
        if movie_id in self.movie_details_cache:
            return self.movie_details_cache[movie_id]
        
        details = self._make_api_request(f"movie/{movie_id}")
        if details:
            self.movie_details_cache[movie_id] = details
            return details
        return {}

    def search_movies(self, query: str, limit: int = 20) -> List[Dict]:
        data = self._make_api_request("search/movie", {"query": query})
        if data and "results" in data:
            return data["results"][:limit]
        return []

    def _extract_movie_features(self, movie_details: Dict) -> Dict:
        features = {}
        movie_genres = [g["name"] for g in movie_details.get("genres", [])]
        for genre in self.all_genres:
            genre_key = genre.lower().replace(' ', '_')
            features[f"genre_{genre_key}"] = 1 if genre in movie_genres else 0
        
        features["vote_average"] = movie_details.get("vote_average", 0.0)
        features["vote_count"] = movie_details.get("vote_count", 0)
        
        release_date_str = movie_details.get("release_date", "")
        years_since_release = 0
        if release_date_str:
            try:
                release_year = int(release_date_str.split('-')[0])
                current_year = datetime.datetime.now().year
                years_since_release = current_year - release_year
            except (ValueError, IndexError):
                pass
        features["years_since_release"] = years_since_release
        
        features["budget"] = movie_details.get("budget", 0)
        features["revenue"] = movie_details.get("revenue", 0)
        features["runtime"] = movie_details.get("runtime", 0)
        
        return features

    def build_user_profile(self, liked_movie_ids: List[int], force_rebuild: bool = False) -> Dict:
        start_time = time.time()
        if not force_rebuild and self.user_profile is not None:
            return self.user_profile

        send_progress("[PERFIL] Construindo perfil do usuário de filmes...")
        self.liked_movie_ids_history = liked_movie_ids
        feature_vectors, liked_movie_names, total_vote_average = [], [], 0.0

        for movie_id in liked_movie_ids:
            details = self.get_movie_details(movie_id)
            if details:
                features = self._extract_movie_features(details)
                feature_vectors.append(list(features.values()))
                liked_movie_names.append(details.get("title", "Filme Desconhecido"))
                total_vote_average += details.get("vote_average", 0.0)
            time.sleep(0.05)

        if not feature_vectors: raise ValueError("Não foi possível obter características dos filmes curtidos.")

        raw_feature_matrix = np.array(feature_vectors)
        vote_counts = raw_feature_matrix[:, len(self.all_genres) + 1]
        with np.errstate(divide='ignore', invalid='ignore'): movie_importances = 1 / np.log1p(vote_counts)
        movie_importances[np.isinf(movie_importances)] = 1
        movie_importances[np.isnan(movie_importances)] = 1
        if np.sum(movie_importances) > 0: movie_importances /= np.sum(movie_importances)
        else: movie_importances = np.ones(len(liked_movie_ids)) / len(liked_movie_ids)

        raw_user_avg_features = np.average(raw_feature_matrix, axis=0, weights=movie_importances)
        feature_matrix = np.array(feature_vectors)
        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, feature_matrix.shape[1]))

        if continuous_indices and raw_feature_matrix.shape[0] > 1:
            self.scaler.fit(raw_feature_matrix[:, continuous_indices])
            feature_matrix[:, continuous_indices] = self.scaler.transform(raw_feature_matrix[:, continuous_indices])

        self.user_features_vector = np.average(feature_matrix, axis=0, weights=movie_importances)
        all_feature_keys = list(self._extract_movie_features(self.get_movie_details(liked_movie_ids[0])).keys()) if liked_movie_ids else []
        user_genre_preferences = {g_name: float(raw_user_avg_features[all_feature_keys.index(f"genre_{g_name.lower().replace(' ', '_')}")]) for g_name in self.all_genres if f"genre_{g_name.lower().replace(' ', '_')}" in all_feature_keys}

        self.user_profile = {
            "liked_movie_names": liked_movie_names, "total_movies_analyzed": len(liked_movie_ids),
            "avg_features": {key: float(raw_user_avg_features[i]) for i, key in enumerate(all_feature_keys)},
            "avg_vote_average": (total_vote_average / len(liked_movie_ids)) if liked_movie_ids else 0.0,
            "genre_preferences": user_genre_preferences
        }
        send_progress(f"✓ Perfil construído em {time.time() - start_time:.2f}s.")
        return self.user_profile

    def _filter_collection_duplicates(self, movies: List[Dict], previously_recommended_ids: set) -> List[Dict]:
        collection_counts: Dict[int, int] = {}
        filtered_movies: List[Dict] = []
        processed_ids = set(self.liked_movie_ids_history).union(previously_recommended_ids)

        for movie in movies:
            if movie['id'] in processed_ids: continue
            details = self.get_movie_details(movie['id'])
            if details and details.get("belongs_to_collection") and isinstance(details.get("belongs_to_collection"), dict):
                collection_id = details["belongs_to_collection"]["id"]
                collection_counts[collection_id] = collection_counts.get(collection_id, 0) + 1
                if collection_counts[collection_id] <= 2:
                    filtered_movies.append(movie)
                    processed_ids.add(movie['id'])
            else:
                filtered_movies.append(movie)
                processed_ids.add(movie['id'])
        return filtered_movies

    def calculate_content_based_similarity(self, candidate_movies: List[Dict]) -> List[Tuple[Dict, float]]:
        start_time = time.time()
        if self.user_features_vector is None: raise ValueError("Perfil do usuário não foi construído.")
        if not self.liked_movie_ids_history: return []

        send_progress("[SIMILARIDADE] Calculando similaridades...")
        movie_vectors, valid_movies = [], []
        profile_feature_keys = list(self.user_profile["avg_features"].keys())

        for movie in candidate_movies:
            details = self.get_movie_details(movie["id"])
            if details:
                features = self._extract_movie_features(details)
                movie_vectors.append([features.get(key, 0) for key in profile_feature_keys])
                valid_movies.append(movie)

        if not movie_vectors: return []

        movie_matrix = np.array(movie_vectors)
        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, movie_matrix.shape[1]))
        
        if continuous_indices and movie_matrix.shape[0] > 0:
            try: movie_matrix[:, continuous_indices] = self.scaler.transform(movie_matrix[:, continuous_indices])
            except Exception: pass

        weighted_user_vector = self.user_features_vector * self.feature_weights
        weighted_movie_matrix = movie_matrix * self.feature_weights
        similarities = cosine_similarity(weighted_user_vector.reshape(1, -1), weighted_movie_matrix)[0]
        
        vote_counts = movie_matrix[:, num_genres + 1]
        with np.errstate(divide='ignore', invalid='ignore'): novelty_scores = 1 / np.log1p(vote_counts)
        novelty_scores[np.isinf(novelty_scores)] = 0
        novelty_scores[np.isnan(novelty_scores)] = 0
        if (novelty_scores.max() - novelty_scores.min()) > 0:
            novelty_scores = (novelty_scores - novelty_scores.min()) / (novelty_scores.max() - novelty_scores.min())

        final_scores = (similarities * 0.9) + (novelty_scores * 0.1)
        send_progress(f"✓ Similaridades e novidade calculadas em {time.time() - start_time:.2f}s.")
        return sorted(list(zip(valid_movies, final_scores)), key=lambda x: x[1], reverse=True)

    def _generate_recommendation_reason(self, movie_details: Dict, similarity: float, context: str = "normal") -> str:
        reasons = []
        if self.user_profile:
            movie_genres = [g["name"] for g in movie_details.get("genres", [])]
            user_genre_prefs = self.user_profile.get("genre_preferences", {})
            sorted_genres = sorted(user_genre_prefs.items(), key=lambda x: x[1], reverse=True)
            for genre, score in sorted_genres[:3]:
                if score > 0.2 and genre in movie_genres: reasons.append(f"gênero {genre}")
        
        if context == "recent": base_explanation = "Um lançamento recente que combina com seu gosto." 
        elif context == "classic": base_explanation = "Um clássico que tem a ver com seu perfil."
        elif context == "diverse": base_explanation = "Para expandir seus horizontes."
        else:
            if similarity >= 0.8: base_explanation = "Este filme combina perfeitamente com seu perfil!"
            elif similarity >= 0.7: base_explanation = "Este filme combina bem com suas preferências."
            else: base_explanation = "Este filme tem algumas características que você gosta."
        
        if reasons: return f"{base_explanation} Principalmente pelo {', '.join(reasons)}."
        return base_explanation

    def _format_recommendation(self, rank: int, movie_details: Dict, similarity: float, context: str) -> Dict:
        return {
            "rank": rank, "similarity_score": float(similarity), "id": movie_details.get("id"),
            "title": movie_details.get("title"), "overview": movie_details.get("overview"),
            "poster_path": f"https://image.tmdb.org/t/p/w500{movie_details.get('poster_path')}" if movie_details.get('poster_path') else None,
            "release_date": movie_details.get("release_date"), "genres": [g["name"] for g in movie_details.get("genres", [])],
            "vote_average": movie_details.get("vote_average"), "url": f"https://www.themoviedb.org/movie/{movie_details.get('id')}",
            "recommendation_reason": self._generate_recommendation_reason(movie_details, similarity, context=context)
        }

    def get_recommendations(self, search_queries: List[str], num_recommendations: int = 5) -> List[Dict]:
        start_time = time.time()
        send_progress("[PRINCIPAIS] Coletando filmes candidatos...")
        all_candidate_movies, previously_recommended_ids = [], set(session.get('recommended_ids', []))
        processed_movie_ids = set(self.liked_movie_ids_history).union(previously_recommended_ids)
        random_pages = random.sample(range(1, 15), 3)
        for page in random_pages:
            for endpoint in ["movie/popular", "movie/top_rated"]:
                data = self._make_api_request(endpoint, {"page": page})
                if data and "results" in data:
                    for movie in data["results"]:
                        if movie["id"] not in processed_movie_ids:
                            all_candidate_movies.append(movie)
                            processed_movie_ids.add(movie["id"])
        for query in search_queries:
            movies = self.search_movies(query, limit=10)
            for movie in movies:
                if movie["id"] not in processed_movie_ids:
                    all_candidate_movies.append(movie)
                    processed_movie_ids.add(movie["id"])
        
        unique_candidate_movies_list = self._filter_collection_duplicates(all_candidate_movies, previously_recommended_ids)
        movie_similarities = self.calculate_content_based_similarity(unique_candidate_movies_list)
        
        recommendations = []
        for movie, similarity in movie_similarities:
            if len(recommendations) >= num_recommendations: break
            details = self.get_movie_details(movie["id"])
            if details:
                recommendations.append(self._format_recommendation(len(recommendations) + 1, details, similarity, "normal"))

        send_progress(f"✓ {len(recommendations)} recomendações principais geradas em {time.time() - start_time:.2f}s.")
        return recommendations

    def get_recent_recommendations(self, num_recs: int = 3) -> List[Dict]:
        start_time = time.time()
        send_progress("[RECENTES] Coletando e ranqueando lançamentos...")
        candidate_movies, previously_recommended_ids = [], set(session.get('recommended_ids', []))
        processed_ids = set(self.liked_movie_ids_history).union(previously_recommended_ids)
        random_pages = random.sample(range(1, 20), 5)
        for page in random_pages:
            data = self._make_api_request("discover/movie", {"sort_by": "popularity.desc", "primary_release_date.lte": datetime.date.today().strftime("%Y-%m-%d"), "primary_release_date.gte": (datetime.date.today() - datetime.timedelta(days=365*2)).strftime("%Y-%m-%d"), "vote_count.gte": 100, "page": page})
            if data and "results" in data:
                for movie in data["results"]:
                    if movie["id"] not in processed_ids:
                        candidate_movies.append(movie)
                        processed_ids.add(movie["id"])
        unique_candidates = self._filter_collection_duplicates(candidate_movies, previously_recommended_ids)
        movie_similarities = self.calculate_content_based_similarity(unique_candidates)
        
        recommendations = []
        for movie, similarity in movie_similarities:
            if len(recommendations) >= num_recs: break
            details = self.get_movie_details(movie["id"])
            if details:
                recommendations.append(self._format_recommendation(len(recommendations) + 1, details, similarity, "recent"))

        send_progress(f"✓ {len(recommendations)} recomendações de recentes geradas em {time.time() - start_time:.2f}s.")
        return recommendations

    def get_classic_recommendations(self, num_recs: int = 3) -> List[Dict]:
        start_time = time.time()
        send_progress("[CLÁSSICOS] Coletando e ranqueando clássicos...")
        candidate_movies, previously_recommended_ids = [], set(session.get('recommended_ids', []))
        processed_ids = set(self.liked_movie_ids_history).union(previously_recommended_ids)
        random_pages = random.sample(range(1, 25), 7)
        for page in random_pages:
            data = self._make_api_request("discover/movie", {"sort_by": "vote_average.desc", "primary_release_date.lte": (datetime.date.today() - datetime.timedelta(days=365*20)).strftime("%Y-%m-%d"), "vote_count.gte": 500, "page": page})
            if data and "results" in data:
                for movie in data["results"]:
                    if movie["id"] not in processed_ids:
                        candidate_movies.append(movie)
                        processed_ids.add(movie["id"])
        unique_candidates = self._filter_collection_duplicates(candidate_movies, previously_recommended_ids)
        movie_similarities = self.calculate_content_based_similarity(unique_candidates)
        
        recommendations = []
        for movie, similarity in movie_similarities:
            if len(recommendations) >= num_recs: break
            details = self.get_movie_details(movie["id"])
            if details:
                recommendations.append(self._format_recommendation(len(recommendations) + 1, details, similarity, "classic"))

        send_progress(f"✓ {len(recommendations)} recomendações de clássicos geradas em {time.time() - start_time:.2f}s.")
        return recommendations

    def get_diverse_genres_recommendations(self, num_recs: int = 3) -> List[Dict]:
        start_time = time.time()
        send_progress("[NOVOS GÊNEROS] Buscando pontes para novos gêneros...")
        if not self.user_profile: return []

        user_prefs = self.user_profile.get("genre_preferences", {})
        sorted_user_genres = [g for g, s in sorted(user_prefs.items(), key=lambda item: item[1], reverse=True) if s > 0.2]
        top_user_genres_ids = [str(self.genre_id_map[g_name]) for g_name in sorted_user_genres[:3] if g_name in self.genre_id_map]
        
        low_preference_genres = [(g_name, self.genre_id_map[g_name]) for g_name in self.all_genres if user_prefs.get(g_name, 0) < 0.1 and g_name in self.genre_id_map]
        if not low_preference_genres or not top_user_genres_ids:
            send_progress("! Não foi possível encontrar gêneros para a ponte.")
            return []

        candidate_movies, used_genres = [], set()
        previously_recommended_ids = set(session.get('recommended_ids', []))
        processed_ids = set(self.liked_movie_ids_history).union(previously_recommended_ids)

        for genre_name, genre_id in random.sample(low_preference_genres, len(low_preference_genres)):
            if len(candidate_movies) > 50: break
            if genre_name in used_genres: continue
            
            data = self._make_api_request("discover/movie", {
                "with_genres": f"{genre_id},{'|'.join(top_user_genres_ids)}",
                "sort_by": "vote_average.desc", "vote_count.gte": 500, "page": random.randint(1, 5)
            })
            if data and "results" in data:
                for movie in data["results"]:
                    if movie["id"] not in processed_ids:
                        candidate_movies.append(movie)
                        processed_ids.add(movie["id"])
        
        unique_candidates = self._filter_collection_duplicates(candidate_movies, previously_recommended_ids)
        movie_similarities = self.calculate_content_based_similarity(unique_candidates)
        
        recommendations = []
        for movie, similarity in movie_similarities:
            if len(recommendations) >= num_recs: break
            details = self.get_movie_details(movie["id"])
            if details:
                rec = self._format_recommendation(len(recommendations) + 1, details, similarity, "diverse")
                movie_genres_set = {g['name'] for g in details.get("genres", [])}
                new_genre_found = list(movie_genres_set.intersection(set([g[0] for g in low_preference_genres])))
                if new_genre_found:
                    rec["recommendation_reason"] = f"Uma ponte para o gênero '{new_genre_found[0]}' que combina com seu perfil."
                    recommendations.append(rec)

        send_progress(f"✓ {len(recommendations)} recomendações de novos gêneros geradas em {time.time() - start_time:.2f}s.")
        return recommendations

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24) 

recommender: Optional[MovieRecommendationSystem] = None
recommendations_data: Dict = {}

@app.route('/')
def input_page():
    global recommendations_data, recommender
    recommendations_data = {}
    recommender = None
    session.pop('recommended_ids', None)
    return render_template('input.html')

@app.route('/api/progress_stream')
def progress_stream():
    def generate():
        while True:
            if progress_queue:
                message = progress_queue.pop(0)
                yield f"data: {json.dumps({'message': message})}\n\n"
                if "---" in message or "Erro" in message:
                    break
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/search')
def api_search_movies():
    query = request.args.get('query')
    if not query: return jsonify({"error": "Query parameter is required"}), 400
    
    global recommender
    if recommender is None:
        TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "6c5c4e771f8ade05ac3f77c826ad6f90")
        recommender = MovieRecommendationSystem(TMDB_API_KEY)
    
    results = recommender.search_movies(query, limit=10)
    return jsonify({"results": results})

@app.route('/api/generate_recommendations', methods=['POST'])
def api_generate_recommendations():
    start_total_time = time.time()
    progress_queue.clear()
    data = request.get_json()
    liked_movie_ids = data.get('movie_ids')

    if not liked_movie_ids or len(liked_movie_ids) < 3:
        return jsonify({"success": False, "error": "Selecione pelo menos 3 filmes."}), 400

    global recommender
    if recommender is None:
        TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "6c5c4e771f8ade05ac3f77c826ad6f90")
        recommender = MovieRecommendationSystem(TMDB_API_KEY)

    try:
        recommender.build_user_profile(liked_movie_ids, force_rebuild=True)
        liked_movies_details = [recommender.get_movie_details(mid) for mid in liked_movie_ids]
        search_queries = [details['title'] for details in liked_movies_details if 'title' in details]

        main_recommendations = recommender.get_recommendations(search_queries, num_recommendations=5)
        recent_recommendations = recommender.get_recent_recommendations(num_recs=3)
        classic_recommendations = recommender.get_classic_recommendations(num_recs=3)
        diverse_genres_recommendations = recommender.get_diverse_genres_recommendations(num_recs=3)

        global recommendations_data
        recommendations_data = {
            "normal": main_recommendations, "recent": recent_recommendations,
            "classic": classic_recommendations, "diverse_genres": diverse_genres_recommendations,
        }
        
        all_rec_ids = [rec['id'] for rec_list in recommendations_data.values() for rec in rec_list if 'id' in rec]
        current_recommended_ids = session.get('recommended_ids', [])
        session['recommended_ids'] = list(set(current_recommended_ids + all_rec_ids))

        send_progress(f"--- Todas as recomendações geradas em {time.time() - start_total_time:.2f}s ---")
        return jsonify({"success": True})
    except Exception as e:
        error_msg = f"Erro ao gerar recomendações: {e}"
        send_progress(error_msg)
        traceback.print_exc()
        return jsonify({"success": False, "error": "Erro interno ao gerar recomendações."}), 500

@app.route('/recommendations')
def recommendations_page():
    global recommendations_data, recommender
    if not recommendations_data or recommender is None or recommender.user_profile is None:
        return redirect(url_for('input_page'))
    
    return render_template('index.html', profile=recommender.user_profile, recommendations=recommendations_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
