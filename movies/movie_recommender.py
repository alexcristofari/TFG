import requests
import numpy as np
from numpy import linalg as np_linalg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import time
import datetime
import os
from flask import Flask, render_template, jsonify, request, redirect, url_for

class MovieRecommendationSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.scaler = MinMaxScaler()
        self.user_profile = None
        self.user_features_vector = None
        self.all_genres = [] # Será preenchido com gêneros do TMDb
        self.movie_details_cache = {}
        self.get_genres() # Popula self.all_genres na inicialização

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        url = f"{self.base_url}/{endpoint}"
        default_params = {"api_key": self.api_key, "language": "pt-BR"}
        if params:
            default_params.update(params)
        
        try:
            response = requests.get(url, params=default_params)
            response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins (4xx ou 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição à API do TMDb: {e}")
            return None

    def get_genres(self) -> List[Dict]:
        data = self._make_api_request("genre/movie/list")
        if data and "genres" in data:
            self.all_genres = [g["name"] for g in data["genres"]]
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

        # Gêneros
        movie_genres = [g["name"] for g in movie_details.get("genres", [])]
        for genre in self.all_genres:
            features[f"genre_{genre.lower().replace(' ', '_')}"] = 1 if genre in movie_genres else 0

        # Popularidade (usando vote_average e vote_count)
        vote_average = movie_details.get("vote_average", 0.0)
        vote_count = movie_details.get("vote_count", 0)
        features["vote_average"] = vote_average
        features["vote_count"] = vote_count

        # Ano de lançamento
        release_date_str = movie_details.get("release_date", "")
        years_since_release = 0
        if release_date_str:
            try:
                release_year = int(release_date_str.split('-')[0])
                current_year = datetime.datetime.now().year
                years_since_release = current_year - release_year
            except ValueError:
                pass
        features["years_since_release"] = years_since_release

        # Orçamento e Receita (se disponíveis e relevantes)
        features["budget"] = movie_details.get("budget", 0)
        features["revenue"] = movie_details.get("revenue", 0)

        # Duração (runtime)
        features["runtime"] = movie_details.get("runtime", 0)

        return features

    def build_user_profile(self, liked_movie_ids: List[int], force_rebuild: bool = False) -> Dict:
        if not force_rebuild and self.user_profile is not None:
            print("Usando perfil existente...")
            return self.user_profile

        print("Construindo perfil do usuário de filmes...")
        feature_vectors = []
        for movie_id in liked_movie_ids:
            details = self.get_movie_details(movie_id)
            if details:
                features = self._extract_movie_features(details)
                feature_vectors.append(list(features.values()))
            time.sleep(0.05) # Pequeno delay para evitar sobrecarregar a API

        if not feature_vectors:
            raise ValueError("Não foi possível obter características dos filmes curtidos.")

        feature_matrix = np.array(feature_vectors)

        # Normaliza as características contínuas
        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, feature_matrix.shape[1]))

        # Calcular as médias das características ANTES da normalização para exibição
        raw_feature_matrix = np.array(feature_vectors)
        raw_user_avg_features = np.mean(raw_feature_matrix, axis=0)

        # Normalizar as características contínuas para o cálculo de similaridade
        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, feature_matrix.shape[1]))

        if continuous_indices:
            # Fit e transform no scaler apenas se houver dados e índices contínuos
            if raw_feature_matrix.shape[0] > 0:
                self.scaler.fit(raw_feature_matrix[:, continuous_indices]) # Fit no raw_feature_matrix
                feature_matrix[:, continuous_indices] = self.scaler.transform(raw_feature_matrix[:, continuous_indices])
            else:
                # Se não houver filmes, o scaler não pode ser ajustado, então o vetor de features será zero
                feature_matrix = np.zeros_like(raw_feature_matrix)
        else:
            # Se não houver continuous_indices, feature_matrix é o mesmo que raw_feature_matrix
            feature_matrix = raw_feature_matrix

        # O user_features_vector é o vetor médio ESCALADO, usado para cálculo de similaridade
        self.user_features_vector = np.mean(feature_matrix, axis=0)

        # Reconstruir o dicionário de features para o perfil do usuário (usando valores RAW para exibição)
        all_feature_keys = list(self._extract_movie_features(self.get_movie_details(liked_movie_ids[0])).keys()) if liked_movie_ids else []
        user_genre_preferences = {}
        for genre in self.all_genres:
            genre_key = f"genre_{genre.lower().replace(' ', '_')}"
            if genre_key in all_feature_keys:
                user_genre_preferences[genre] = float(raw_user_avg_features[all_feature_keys.index(genre_key)])

        self.user_profile = {
            "total_movies_analyzed": len(liked_movie_ids),
            "avg_features": {key: float(raw_user_avg_features[i]) for i, key in enumerate(all_feature_keys)},
            "genre_preferences": user_genre_preferences
        }
        print("Perfil do usuário de filmes construído com sucesso.")
        return self.user_profile

    def calculate_content_based_similarity(self, candidate_movies: List[Dict]) -> List[Tuple[Dict, float]]:
        if self.user_features_vector is None:
            raise ValueError("Perfil do usuário não foi construído. Execute build_user_profile() primeiro.")

        print("Calculando similaridades baseadas em conteúdo para filmes...")
        movie_vectors = []
        valid_movies = []
        
        user_profile_feature_keys = list(self.user_profile["avg_features"].keys())

        for movie in candidate_movies:
            details = self.get_movie_details(movie["id"])
            if details:
                features = self._extract_movie_features(details)
                current_movie_vector = [features.get(key, 0) for key in user_profile_feature_keys]
                
                if len(current_movie_vector) == len(self.user_features_vector):
                    movie_vectors.append(current_movie_vector)
                    valid_movies.append(movie)
                else:
                    print(f"Aviso: Vetor de features do filme {movie.get('title', 'Desconhecido')} tem tamanho diferente do perfil do usuário. Ignorando.")

        if not movie_vectors:
            print("Nenhum vetor de características válido encontrado para filmes candidatos!")
            return []

        movie_matrix = np.array(movie_vectors)

        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, movie_matrix.shape[1]))
        if continuous_indices and len(continuous_indices) > 0 and movie_matrix.shape[0] > 0:
            try:
                movie_matrix[:, continuous_indices] = self.scaler.transform(movie_matrix[:, continuous_indices])
            except Exception as e:
                print(f"Erro na normalização de filmes candidatos: {e}")
                pass

        user_vector = self.user_features_vector.reshape(1, -1)

        if user_vector.shape[1] != movie_matrix.shape[1]:
            print(f"Dimensões incompatíveis: usuário {user_vector.shape}, filmes {movie_matrix.shape}")
            return []

        similarities = cosine_similarity(user_vector, movie_matrix)[0]

        movie_similarities = list(zip(valid_movies, similarities))
        movie_similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"✓ Similaridades calculadas para {len(valid_movies)} filmes.")
        return movie_similarities

    def get_recommendations(self, liked_movie_ids: List[int], search_queries: List[str], num_recommendations: int = 10) -> List[Dict]:
        self.build_user_profile(liked_movie_ids, force_rebuild=True)

        all_candidate_movies = []
        processed_movie_ids = set()

        # 1. Adicionar filmes populares (para diversidade e filmes atuais)
        for page in range(1, 3): # Buscar de 2 páginas de populares
            popular_movies = self._make_api_request("movie/popular", {"page": page})
            if popular_movies and "results" in popular_movies:
                for movie in popular_movies["results"]:
                    if movie["id"] not in processed_movie_ids:
                        all_candidate_movies.append(movie)
                        processed_movie_ids.add(movie["id"])

        # 2. Adicionar filmes bem avaliados (para qualidade e clássicos)
        for page in range(1, 3): # Buscar de 2 páginas de bem avaliados
            top_rated_movies = self._make_api_request("movie/top_rated", {"page": page})
            if top_rated_movies and "results" in top_rated_movies:
                for movie in top_rated_movies["results"]:
                    if movie["id"] not in processed_movie_ids:
                        all_candidate_movies.append(movie)
                        processed_movie_ids.add(movie["id"])

        # 3. Adicionar filmes baseados em queries de busca (para cobrir interesses específicos)
        for query in search_queries:
            movies = self.search_movies(query, limit=10) # Limitar a 10 por query para não sobrecarregar
            for movie in movies:
                if movie["id"] not in processed_movie_ids:
                    all_candidate_movies.append(movie)
             # Remover duplicatas e filmes já curtidos da lista final de candidatos
        final_candidate_movies = []
        for movie in all_candidate_movies:
            if movie["id"] not in liked_movie_ids and movie["id"] not in processed_movie_ids:
                final_candidate_movies.append(movie)
                processed_movie_ids.add(movie["id"]) # Adicionar ao set para evitar duplicatas na lista final
        
        # Garantir que processed_movie_ids inclua todos os filmes já processados para evitar re-adição
        # (já feito nos loops anteriores, mas reforçando a lógica)
        
        # A lista de candidatos para cálculo de similaridade será final_candidate_movies
        unique_candidate_movies_list = final_candidate_movies     

        print(f"✓ Total de filmes candidatos únicos: {len(unique_candidate_movies_list)}")

        movie_similarities = self.calculate_content_based_similarity(unique_candidate_movies_list)

        recommendations = []
        for movie, similarity in movie_similarities[:num_recommendations]:
            details = self.get_movie_details(movie["id"])
            if details:
                recommendation = {
                    "rank": len(recommendations) + 1,
                    "similarity_score": float(similarity),
                    "movie_info": {
                        "id": movie["id"],
                        "title": details.get("title"),
                        "overview": details.get("overview"),
                        "poster_path": f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}" if details.get('poster_path') else None,
                        "release_date": details.get("release_date"),
                        "genres": [g["name"] for g in details.get("genres", [])],
                        "vote_average": details.get("vote_average"),
                        "url": f"https://www.themoviedb.org/movie/{movie['id']}"
                    },
                    "recommendation_reason": self._generate_recommendation_reason(details, similarity)
                }
                recommendations.append(recommendation)
        
        print(f"✓ {len(recommendations)} recomendações de filmes geradas.")
        return recommendations

    def _generate_recommendation_reason(self, movie_details: Dict, similarity: float) -> str:
        user_features = self.user_profile["avg_features"]
        movie_features = self._extract_movie_features(movie_details)

        reasons = []

        # Gêneros mais proeminentes no perfil do usuário
        user_genre_preferences = {genre: user_features[f"genre_{genre.lower().replace(' ', '_')}"] for genre in self.all_genres if f"genre_{genre.lower().replace(' ', '_')}" in user_features}
        sorted_genres = sorted(user_genre_preferences.items(), key=lambda item: item[1], reverse=True)
        
        # Adicionar os 3 gêneros mais fortes do perfil do usuário que também estão no filme
        for genre, score in sorted_genres[:3]: # Top 3 gêneros do perfil
            genre_key = f"genre_{genre.lower().replace(' ', '_')}"
            if genre_key in movie_features and movie_features[genre_key] > 0: # Se o filme tem esse gênero
                reasons.append(f"gênero {genre}")
        
        if movie_features["vote_count"] > 10000 and movie_features["vote_average"] > 7.0:
            reasons.append("muito popular e bem avaliado")
        elif movie_features["vote_count"] > 1000 and movie_features["vote_average"] > 6.0:
            reasons.append("popular")

        if movie_features["years_since_release"] < 2:
            reasons.append("lançamento recente")
        elif movie_features["years_since_release"] > 10:
            reasons.append("clássico")

        if similarity >= 0.8:
            similarity_class = "Excelente combinação"
            explanation = "Este filme combina perfeitamente com seu perfil!"
        elif similarity >= 0.6:
            similarity_class = "Boa combinação"
            explanation = "Este filme combina bem com suas preferências."
        elif similarity >= 0.4:
            similarity_class = "Combinação moderada"
            explanation = "Este filme tem algumas características que você gosta."
        else:
            similarity_class = "Combinação fraca"
            explanation = "Este filme pode ser interessante, mas com menor similaridade."

        if reasons:
            return f"{explanation} Você provavelmente vai gostar por causa do {' '.join(reasons)}."
        else:
            return f"{explanation} Baseado em características gerais."

# Flask App
app = Flask(__name__, template_folder='templates')

recommender = None
recommendations_data = []

@app.route('/')
def input_page():
    return render_template('input.html')

@app.route('/api/search')
def api_search_movies():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    global recommender
    if recommender is None:
        TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "6c5c4e771f8ade05ac3f77c826ad6f90")
        recommender = MovieRecommendationSystem(TMDB_API_KEY)

    results = recommender.search_movies(query, limit=10)
    formatted_results = []
    for movie in results:
        formatted_results.append({
            "id": movie["id"],
            "title": movie["title"],
            "release_date": movie.get("release_date", ""),
            "poster_path": movie.get("poster_path"),
            "overview": movie.get("overview", "")
        })
    return jsonify({"results": formatted_results})

@app.route('/api/generate_recommendations', methods=['POST'])
def api_generate_recommendations():
    data = request.get_json()
    liked_movie_ids = data.get('movie_ids')

    if not liked_movie_ids or len(liked_movie_ids) < 3:
        return jsonify({"success": False, "error": "Selecione pelo menos 3 filmes."}), 400

    try:
        global recommender, recommendations_data
        if recommender is None:
            TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "6c5c4e771f8ade05ac3f77c826ad6f90")
            recommender = MovieRecommendationSystem(TMDB_API_KEY)

        search_queries = [
            "ação", "drama", "ficção científica", "aventura", "comédia",
            "suspense", "terror", "fantasia", "romance", "documentário"
        ]
        recommendations_data = recommender.get_recommendations(liked_movie_ids, search_queries, num_recommendations=10)
        
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "Erro interno do servidor: " + str(e)}), 500

@app.route('/recommendations')
def show_recommendations():
    if recommender and recommender.user_profile and recommendations_data:
        return render_template('index.html', 
                             profile=recommender.user_profile,
                             recommendations=recommendations_data,
                             all_genres=recommender.all_genres)
    else:
        return redirect(url_for('input_page'))

def main():
    global recommender
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "6c5c4e771f8ade05ac3f77c826ad6f90")
    recommender = MovieRecommendationSystem(TMDB_API_KEY)

    print("\nIniciando interface web de filmes...")
    print("Acesse http://127.0.0.1:5001 no seu navegador para informar seus filmes curtidos.")
    
    app.run(debug=False, port=5001)

if __name__ == "__main__":
    main()
