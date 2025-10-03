
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
        # Identificar índices das características contínuas para normalização
        # Assumindo que as primeiras características são gêneros (binárias) e as seguintes são contínuas
        # genre_... (binárias), vote_average, vote_count, years_since_release, budget, revenue, runtime
        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, feature_matrix.shape[1]))

        if continuous_indices:
            self.scaler.fit(feature_matrix[:, continuous_indices])
            feature_matrix[:, continuous_indices] = self.scaler.transform(feature_matrix[:, continuous_indices])

        user_avg_features = np.mean(feature_matrix, axis=0)
        self.user_features_vector = user_avg_features

        # Reconstruir o dicionário de features para o perfil do usuário
        all_feature_keys = list(self._extract_movie_features(self.get_movie_details(liked_movie_ids[0])).keys()) if liked_movie_ids else []
        self.user_profile = {
            "total_movies_analyzed": len(liked_movie_ids),
            "avg_features": {key: float(user_avg_features[i]) for i, key in enumerate(all_feature_keys)}
        }
        print("Perfil do usuário de filmes construído com sucesso.")
        return self.user_profile

    def calculate_content_based_similarity(self, candidate_movies: List[Dict]) -> List[Tuple[Dict, float]]:
        if self.user_features_vector is None:
            raise ValueError("Perfil do usuário não foi construído. Execute build_user_profile() primeiro.")

        print("Calculando similaridades baseadas em conteúdo para filmes...")
        movie_vectors = []
        valid_movies = []
        
        # Garantir que todos os vetores de features tenham o mesmo tamanho do vetor do usuário
        # Para isso, extraímos as chaves do primeiro filme do perfil do usuário para garantir a ordem
        user_profile_feature_keys = list(self.user_profile["avg_features"].keys())

        for movie in candidate_movies:
            details = self.get_movie_details(movie["id"])
            if details:
                features = self._extract_movie_features(details)
                # Criar vetor de features na mesma ordem do perfil do usuário
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

        # Normaliza as características contínuas dos filmes candidatos usando o scaler do perfil do usuário
        num_genres = len(self.all_genres)
        continuous_indices = list(range(num_genres, movie_matrix.shape[1]))
        if continuous_indices and len(continuous_indices) > 0 and movie_matrix.shape[0] > 0:
            try:
                movie_matrix[:, continuous_indices] = self.scaler.transform(movie_matrix[:, continuous_indices])
            except Exception as e:
                print(f"Erro na normalização de filmes candidatos: {e}")
                # Se o scaler não foi ajustado corretamente (ex: user_profile vazio), pode falhar
                # Neste caso, podemos pular a normalização para os candidatos ou usar um scaler padrão
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

        print(f"Gerando {num_recommendations} recomendações de filmes...")

        all_candidate_movies = []
        for query in search_queries:
            movies = self.search_movies(query, limit=20) # Busca filmes com base nas queries
            all_candidate_movies.extend(movies)
        
        # Adicionar filmes populares como candidatos para diversificar
        popular_movies = self._make_api_request("movie/popular", {"page": 1})
        if popular_movies and "results" in popular_movies:
            all_candidate_movies.extend(popular_movies["results"])

        # Remover duplicatas e filmes já curtidos
        unique_candidate_movies = {}
        for movie in all_candidate_movies:
            if movie["id"] not in unique_candidate_movies and movie["id"] not in liked_movie_ids:
                unique_candidate_movies[movie["id"]] = movie
        
        unique_candidate_movies_list = list(unique_candidate_movies.values())
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

        # Gêneros
        for genre in self.all_genres:
            genre_key = f"genre_{genre.lower().replace(' ', '_')}"
            if genre_key in user_features and genre_key in movie_features:
                if user_features[genre_key] > 0.5 and movie_features[genre_key] > 0.5:
                    reasons.append(f"gênero {genre}")
        
        # Popularidade
        if movie_features["vote_count"] > 10000 and movie_features["vote_average"] > 7.0:
            reasons.append("muito popular e bem avaliado")
        elif movie_features["vote_count"] > 1000 and movie_features["vote_average"] > 6.0:
            reasons.append("popular")

        # Ano de lançamento
        if movie_features["years_since_release"] < 2:
            reasons.append("lançamento recente")
        elif movie_features["years_since_release"] > 10:
            reasons.append("clássico")

        # Classificação da similaridade
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
            return f"{explanation} Você provavelmente vai gostar por causa do {', '.join(reasons)}."
        else:
            return f"{explanation} Baseado em características gerais."

# Flask App
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Variáveis globais para armazenar o sistema de recomendação e as recomendações
recommender = None
recommendations_data = []

@app.route('/')
def index():
    """Renderiza a página principal com o perfil do usuário e as recomendações"""
    if recommender and recommender.user_profile:
        return render_template('index.html', 
                             profile=recommender.user_profile,
                             recommendations=recommendations_data,
                             all_genres=recommender.all_genres)
    else:
        return "<h1>Erro: Sistema de recomendação não inicializado</h1>"

@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint para obter recomendações em formato JSON"""
    return jsonify(recommendations_data)

@app.route('/api/profile')
def api_profile():
    """API endpoint para obter o perfil do usuário em formato JSON"""
    if recommender and recommender.user_profile:
        return jsonify(recommender.user_profile)
    else:
        return jsonify({"error": "Perfil não disponível"}), 404

def main():
    """Função principal para executar o sistema de recomendação"""
    TMDB_API_KEY = "6c5c4e771f8ade05ac3f77c826ad6f90"
    
    global recommender, recommendations_data
    
    try:
        # Inicializa o sistema de recomendação
        print("=== SISTEMA DE RECOMENDAÇÃO DE FILMES - TMDb ===")
        print("Baseado no TCC: Sistema de Recomendação para Músicas, Filmes e Jogos")
        print()
        
        recommender = MovieRecommendationSystem(TMDB_API_KEY)
        
        # IDs de filmes que o usuário 'curtiu' (exemplo)
        # Estes IDs são de filmes populares para simular um perfil inicial
        liked_movie_ids = [278, 19404, 155, 680, 122, 13, 238, 424, 389, 769]
        # 278: Um Sonho de Liberdade
        # 19404: Dilwale Dulhania Le Jayenge
        # 155: Batman: O Cavaleiro das Trevas
        # 680: Pulp Fiction
        # 122: O Senhor dos Anéis: O Retorno do Rei
        # 13: Forrest Gump
        # 238: O Poderoso Chefão
        # 424: A Lista de Schindler
        # 389: 12 Homens e uma Sentença
        # 769: GoodFellas
        
        # Constrói o perfil do usuário
        user_profile = recommender.build_user_profile(liked_movie_ids, force_rebuild=True)
        
        # Gera recomendações baseadas no perfil geral
        search_queries = [
            "ação",
            "drama",
            "ficção científica",
            "aventura",
            "comédia"
        ]
        
        print(f"\nGerando recomendações baseadas em: {', '.join(search_queries)}")
        recommendations_data = recommender.get_recommendations(liked_movie_ids, search_queries, num_recommendations=10)
        
        # Exibe recomendações
        print("\n=== SUAS RECOMENDAÇÕES DE FILMES PERSONALIZADAS ===")
        for rec in recommendations_data:
            print(f"\n{rec['rank']}. {rec['movie_info']['title']}")
            print(f"   Similaridade: {rec['similarity_score']:.2f}")
            print(f"   Gêneros: {', '.join(rec['movie_info']['genres'])}")
            print(f"   {rec['recommendation_reason']}")
            print(f"   TMDb: {rec['movie_info']['url']}")
        
        print("\n=== SISTEMA DE RECOMENDAÇÃO DE FILMES EXECUTADO COM SUCESSO! ===")
        print("Este exemplo demonstra como implementar o sistema descrito no seu TCC.")
        print("\nIniciando interface web...")
        print("Acesse http://127.0.0.1:5001 no seu navegador para ver as recomendações.")
        
        # Inicia a interface web
        app.run(debug=False, port=5001)
        
    except Exception as e:
        print(f"Erro: {e}")
        print("\nVerifique se:")
        print("1. Sua chave da API do TMDb está correta")
        print("2. Sua conexão com a internet está funcionando")

if __name__ == "__main__":
    main()
