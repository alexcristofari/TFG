"""
Sistema de Recomendação Híbrido para Jogos - Versão Aprimorada
Este módulo implementa um sistema de recomendação que combina filtragem colaborativa
e filtragem baseada em conteúdo usando dados da Steam.
"""

import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import time
import datetime
import os
from flask import Flask, render_template, jsonify

class SteamRecommendationSystem:
    """
    Sistema de Recomendação Híbrido para Jogos usando API da Steam
    
    Implementa os conceitos descritos no TCC:
    - Filtragem baseada em conteúdo usando características dos jogos
    - Perfil do usuário baseado em biblioteca de jogos
    - Normalização de características usando Min-Max
    - Similaridade do cosseno para recomendações
    """
    
    def __init__(self, api_key: str, steam_id: str):
        """
        Inicializa o sistema de recomendação
        
        Args:
            api_key (str): Chave da API da Steam
            steam_id (str): ID do usuário na Steam (formato 64 bits)
        """
        self.api_key = api_key
        self.steam_id = steam_id
        self.base_url = "https://api.steampowered.com"
        
        # Scaler para normalização das características
        self.scaler = MinMaxScaler()
        
        # Cache para armazenar dados
        self.user_profile = None
        self.user_features_vector = None
        self.games_database = []
        self.owned_games_ids = set()
        
        # Cache para requisições à API
        self.app_details_cache = {}
        
    def get_owned_games(self) -> List[Dict]:
        """Obtém a biblioteca de jogos do usuário"""
        url = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/"
        params = {
            'key': self.api_key,
            'steamid': self.steam_id,
            'include_appinfo': True,
            'include_played_free_games': True
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'response' not in data:
                raise ValueError("Resposta inválida da API")
                
            games = data.get('response', {}).get('games', [])
            return games
            
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                raise ValueError("Chave da API inválida ou expirada")
            elif response.status_code == 403:
                raise ValueError("Acesso negado. Verifique se sua biblioteca de jogos está pública")
            elif response.status_code == 400:
                raise ValueError("Requisição inválida. Verifique o formato do Steam ID")
            else:
                raise ValueError(f"Erro HTTP: {http_err}")
        except Exception as err:
            raise ValueError(f"Erro ao obter jogos: {err}")
    
    def get_app_details(self, app_id: int) -> Dict:
        """Obtém detalhes de um jogo específico"""
        if app_id in self.app_details_cache:
            return self.app_details_cache[app_id]
        
        url = "https://store.steampowered.com/api/appdetails"
        params = {
            'appids': app_id,
            'cc': 'br'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if str(app_id) in data and data[str(app_id)]['success']:
                details = data[str(app_id)]['data']
                self.app_details_cache[app_id] = details
                return details
            
            return {}
        except Exception as e:
            print(f"Erro ao obter detalhes do app {app_id}: {e}")
            return {}
    
    def build_user_profile(self, limit: int = 50, force_rebuild: bool = False) -> Dict:
        """
        Constrói o perfil do usuário baseado em sua biblioteca de jogos
        
        Args:
            limit (int): Número de jogos a analisar
            force_rebuild (bool): Se True, força a reconstrução do perfil mesmo se já existir
            
        Returns:
            Dict: Perfil do usuário com características médias
        """
        # Verifica se já temos um perfil salvo e se não for para forçar reconstrução
        if not force_rebuild and self.user_profile is not None:
            print("Usando perfil existente...")
            return self.user_profile
            
        # Tenta carregar do arquivo
        if not force_rebuild and os.path.exists("steam_profile.json"):
            try:
                with open("steam_profile.json", "r", encoding="utf-8") as f:
                    self.user_profile = json.load(f)
                    
                # Verifica se o perfil tem todas as características necessárias
                # Se não tiver, força a reconstrução
                required_features = ['price', 'positive_ratio', 'total_reviews', 
                                   'years_since_release', 'supports_controller']
                for feature in required_features:
                    if feature not in self.user_profile.get('avg_features', {}):
                        print(f"Perfil salvo não contém a característica '{feature}'. Reconstruindo...")
                        force_rebuild = True
                        break
                
                if not force_rebuild:
                    print("Perfil carregado do arquivo...")
                    return self.user_profile
            except (FileNotFoundError, json.JSONDecodeError):
                pass
                
        print("Construindo perfil do usuário...")
        
        # Obtém jogos do usuário
        owned_games = self.get_owned_games()
        
        if not owned_games:
            raise ValueError("Usuário não possui jogos na biblioteca")
        
        # Armazena IDs dos jogos possuídos
        self.owned_games_ids = {game['appid'] for game in owned_games}
        
        # Ordena por tempo de jogo
        sorted_games = sorted(owned_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
        
        # Limita número de jogos
        games_to_analyze = sorted_games[:limit]
        
        # Extrai características dos jogos
        feature_vectors = []
        analyzed_games = []
        
        for game in games_to_analyze:
            app_id = game['appid']
            details = self.get_app_details(app_id)
            
            if not details:
                continue
                
            # Extrai características relevantes
            genres = [g['description'] for g in details.get('genres', [])]
            categories = [c['description'] for c in details.get('categories', [])]
            
            # Características numéricas - TRATAMENTO MELHORADO DO PREÇO
            price_info = details.get('price_overview', {})
            price = price_info.get('final', 0) / 100.0  # Converte centavos para reais
            
            # Calcula a proporção de avaliações positivas
            positive = details.get('positive', 0)
            total = details.get('total', 1)
            positive_ratio = positive / total if total > 0 else 0
            
            # Data de lançamento (anos desde o lançamento)
            release_date = details.get('release_date', {}).get('date', '')
            years_since_release = 0
            if release_date:
                try:
                    # Tenta extrair o ano
                    year = int(release_date.split(',')[-1].strip())
                    current_year = datetime.datetime.now().year
                    years_since_release = current_year - year
                except:
                    pass
            
            # Suporte a controle
            supports_controller = 1 if 'Full Controller Support' in categories else 0
            
            features = {
                'price': price,
                'positive_ratio': positive_ratio,
                'total_reviews': total,  # Número total de reviews (popularidade)
                'years_since_release': years_since_release,
                'supports_controller': supports_controller,
                'is_multiplayer': 1 if 'Multi-player' in categories else 0,
                'is_singleplayer': 1 if 'Single-player' in categories else 0,
                'is_coop': 1 if 'Co-op' in categories else 0,
                'is_rpg': 1 if 'RPG' in genres else 0,
                'is_action': 1 if 'Action' in genres else 0,
                'is_strategy': 1 if 'Strategy' in genres else 0,
                'is_indie': 1 if 'Indie' in genres else 0,
                'is_adventure': 1 if 'Adventure' in genres else 0,
                'is_sports': 1 if 'Sports' in genres else 0,
                'is_racing': 1 if 'Racing' in genres else 0,
                'is_simulation': 1 if 'Simulation' in genres else 0,
                'is_casual': 1 if 'Casual' in genres else 0,
                'playtime': game.get('playtime_forever', 0)  # Em minutos
            }
            
            feature_vectors.append(list(features.values()))
            analyzed_games.append({
                'name': game['name'],
                'appid': app_id,
                'playtime': game.get('playtime_forever', 0),
                'features': features
            })
            
            # Respeitar limites da API
            time.sleep(0.1)
        
        if not feature_vectors:
            raise ValueError("Não foi possível obter características dos jogos")
        
        # Converte para numpy array
        feature_matrix = np.array(feature_vectors)
        
        # Normaliza as características contínuas (preço, positive_ratio, playtime, total_reviews, years_since_release)
        continuous_indices = [0, 1, 2, 3, 17]  # Índices de preço, positive_ratio, total_reviews, years_since_release, playtime
        self.scaler.fit(feature_matrix[:, continuous_indices])
        feature_matrix[:, continuous_indices] = self.scaler.transform(feature_matrix[:, continuous_indices])
        
        # Calcula características médias do usuário
        user_avg_features = np.mean(feature_matrix, axis=0)
        
        # Armazena o perfil do usuário
        self.user_features_vector = user_avg_features
        
        # Cria perfil detalhado
        self.user_profile = {
            'steam_id': self.steam_id,
            'total_games_analyzed': len(analyzed_games),
            'avg_features': {
                'price': float(user_avg_features[0]),
                'positive_ratio': float(user_avg_features[1]),
                'total_reviews': float(user_avg_features[2]),
                'years_since_release': float(user_avg_features[3]),
                'supports_controller': float(user_avg_features[4]),
                'is_multiplayer': float(user_avg_features[5]),
                'is_singleplayer': float(user_avg_features[6]),
                'is_coop': float(user_avg_features[7]),
                'is_rpg': float(user_avg_features[8]),
                'is_action': float(user_avg_features[9]),
                'is_strategy': float(user_avg_features[10]),
                'is_indie': float(user_avg_features[11]),
                'is_adventure': float(user_avg_features[12]),
                'is_sports': float(user_avg_features[13]),
                'is_racing': float(user_avg_features[14]),
                'is_simulation': float(user_avg_features[15]),
                'is_casual': float(user_avg_features[16]),
                'playtime': float(user_avg_features[17])
            },
            'top_games': analyzed_games[:10]
        }
        
        # Salva o perfil automaticamente
        self.save_profile()
        
        print(f"✓ Perfil construído com {len(analyzed_games)} jogos")
        return self.user_profile
    
    def search_games(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Busca jogos na Steam e analisa suas características
        
        Args:
            query (str): Termo de busca
            limit (int): Número de resultados
            
        Returns:
            List[Dict]: Lista de jogos com características analisadas
        """
        print(f"Buscando jogos para: '{query}'...")
        
        # Busca jogos usando a API de busca da Steam
        url = "https://store.steampowered.com/api/storesearch"
        params = {
            'term': query,
            'l': 'portuguese',
            'cc': 'br'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            games = data.get('items', [])[:limit]
            
            analyzed_games = []
            for game in games:
                app_id = game['id']
                details = self.get_app_details(app_id)
                
                if not details:
                    continue
                    
                # Extrai características
                genres = [g['description'] for g in details.get('genres', [])]
                categories = [c['description'] for c in details.get('categories', [])]
                
                # TRATAMENTO MELHORADO DO PREÇO
                price_info = details.get('price_overview', {})
                price = price_info.get('final', 0) / 100.0  # Converte centavos para reais
                
                # Se não tiver preço na detalhe, usa o da busca
                if price == 0 and 'price' in game:
                    price = game.get('price', 0) / 100.0
                
                # Calcula a proporção de avaliações positivas
                positive = details.get('positive', 0)
                total = details.get('total', 1)
                positive_ratio = positive / total if total > 0 else 0
                
                # Data de lançamento (anos desde o lançamento)
                release_date = details.get('release_date', {}).get('date', '')
                years_since_release = 0
                if release_date:
                    try:
                        # Tenta extrair o ano
                        year = int(release_date.split(',')[-1].strip())
                        current_year = datetime.datetime.now().year
                        years_since_release = current_year - year
                    except:
                        pass
                
                # Suporte a controle
                supports_controller = 1 if 'Full Controller Support' in categories else 0
                
                features = {
                    'price': price,
                    'positive_ratio': positive_ratio,
                    'total_reviews': total,  # Número total de reviews (popularidade)
                    'years_since_release': years_since_release,
                    'supports_controller': supports_controller,
                    'is_multiplayer': 1 if 'Multi-player' in categories else 0,
                    'is_singleplayer': 1 if 'Single-player' in categories else 0,
                    'is_coop': 1 if 'Co-op' in categories else 0,
                    'is_rpg': 1 if 'RPG' in genres else 0,
                    'is_action': 1 if 'Action' in genres else 0,
                    'is_strategy': 1 if 'Strategy' in genres else 0,
                    'is_indie': 1 if 'Indie' in genres else 0,
                    'is_adventure': 1 if 'Adventure' in genres else 0,
                    'is_sports': 1 if 'Sports' in genres else 0,
                    'is_racing': 1 if 'Racing' in genres else 0,
                    'is_simulation': 1 if 'Simulation' in genres else 0,
                    'is_casual': 1 if 'Casual' in genres else 0,
                    'playtime': 0  # Não disponível na busca
                }
                
                # Cria vetor de características normalizadas
                feature_vector = list(features.values())
                
                # Normaliza as características contínuas usando o scaler do perfil do usuário
                continuous_indices = [0, 1, 2, 3, 17]  # preço, positive_ratio, total_reviews, years_since_release, playtime
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_array[:, continuous_indices] = self.scaler.transform(feature_array[:, continuous_indices])
                feature_vector = feature_array.flatten().tolist()
                
                analyzed_game = {
                    'id': app_id,
                    'name': game['name'],
                    'price': price,
                    'thumbnail': game.get('tiny_image', ''),
                    'features': features,
                    'feature_vector': feature_vector,
                    'genres': genres,
                    'categories': categories,
                    'url': f"https://store.steampowered.com/app/{app_id}"
                }
                analyzed_games.append(analyzed_game)
                
                # Respeitar limites da API
                time.sleep(0.1)
            
            print(f"✓ Analisados {len(analyzed_games)} jogos")
            return analyzed_games
            
        except Exception as e:
            print(f"Erro na busca de jogos: {e}")
            return []
    
    def calculate_content_based_similarity(self, games: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Calcula similaridade baseada em conteúdo entre o perfil do usuário e os jogos
        
        Args:
            games (List[Dict]): Lista de jogos analisados
            
        Returns:
            List[Tuple[Dict, float]]: Lista de (jogo, similaridade) ordenada por similaridade
        """
        if self.user_features_vector is None:
            raise ValueError("Perfil do usuário não foi construído. Execute build_user_profile() primeiro.")
        
        print("Calculando similaridades baseadas em conteúdo...")
        
        # Extrai vetores de características dos jogos
        game_vectors = np.array([game['feature_vector'] for game in games])
        user_vector = self.user_features_vector.reshape(1, -1)
        
        # Calcula similaridade do cosseno
        similarities = cosine_similarity(user_vector, game_vectors)[0]
        
        # Combina jogos com suas similaridades
        game_similarities = list(zip(games, similarities))
        
        # Ordena por similaridade (maior primeiro)
        game_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✓ Similaridades calculadas para {len(games)} jogos")
        return game_similarities
    
    def get_recommendations(self, search_queries: List[str], num_recommendations: int = 10, 
                           max_per_genre: int = 3, exclude_owned: bool = True) -> List[Dict]:
        """
        Gera recomendações baseadas em múltiplas consultas de busca
        
        Args:
            search_queries (List[str]): Lista de termos de busca
            num_recommendations (int): Número de recomendações a retornar
            max_per_genre (int): Máximo de recomendações por gênero (para diversidade)
            exclude_owned (bool): Se True, exclui jogos que o usuário já possui
            
        Returns:
            List[Dict]: Lista de recomendações ordenadas por relevância
        """
        if self.user_profile is None:
            raise ValueError("Perfil do usuário não foi construído. Execute build_user_profile() primeiro.")
        
        print(f"Gerando recomendações para {len(search_queries)} consultas...")
        
        all_games = []
        
        # Busca e analisa jogos para cada consulta
        for query in search_queries:
            games = self.search_games(query, limit=20)
            all_games.extend(games)
        
        # Remove duplicatas baseadas no ID do jogo
        unique_games = {}
        for game in all_games:
            if game['id'] not in unique_games:
                unique_games[game['id']] = game
        
        unique_games_list = list(unique_games.values())
        print(f"✓ Total de jogos únicos encontrados: {len(unique_games_list)}")
        
        # Se for excluir jogos já possuídos
        if exclude_owned:
            filtered_games = [game for game in unique_games_list if game['id'] not in self.owned_games_ids]
            print(f"✓ Jogos após filtrar os já possuídos: {len(filtered_games)}")
        else:
            filtered_games = unique_games_list
        
        # Calcula similaridades
        game_similarities = self.calculate_content_based_similarity(filtered_games)
        
        # Prepara recomendações finais com diversidade por gênero
        recommendations = []
        genre_count = {}  # Contador por gênero
        
        for game, similarity in game_similarities:
            # Verifica se já atingimos o limite para cada gênero
            genres = game['genres']
            skip = False
            for genre in genres:
                if genre_count.get(genre, 0) >= max_per_genre:
                    skip = True
                    break
            
            if skip:
                continue
                
            # Adiciona a recomendação
            recommendation = {
                'rank': len(recommendations) + 1,
                'similarity_score': float(similarity),
                'game_info': {
                    'name': game['name'],
                    'price': game['price'],
                    'url': game['url'],
                    'thumbnail': game['thumbnail'],
                    'genres': game['genres'],
                    'categories': game['categories']
                },
                'feature_match': {
                    'price': game['features']['price'],
                    'positive_ratio': game['features']['positive_ratio'],
                    'total_reviews': game['features']['total_reviews'],
                    'years_since_release': game['features']['years_since_release'],
                    'supports_controller': game['features']['supports_controller'],
                    'is_multiplayer': game['features']['is_multiplayer'],
                    'is_singleplayer': game['features']['is_singleplayer'],
                    'is_rpg': game['features']['is_rpg'],
                    'is_action': game['features']['is_action'],
                    'is_strategy': game['features']['is_strategy']
                },
                'recommendation_reason': self._generate_recommendation_reason(game, similarity)
            }
            recommendations.append(recommendation)
            
            # Atualiza contador de gêneros
            for genre in genres:
                genre_count[genre] = genre_count.get(genre, 0) + 1
            
            # Verifica se já temos o número desejado de recomendações
            if len(recommendations) >= num_recommendations:
                break
        
        print(f"✓ {len(recommendations)} recomendações geradas")
        return recommendations
    
    def _generate_recommendation_reason(self, game: Dict, similarity: float) -> str:
        """
        Gera uma explicação para a recomendação baseada nas características
        
        Args:
            game (Dict): Informações do jogo
            similarity (float): Score de similaridade
            
        Returns:
            str: Explicação da recomendação
        """
        features = game['features']
        user_features = self.user_profile['avg_features']
        
        reasons = []
        
        # Analisa características principais
        if abs(features['is_multiplayer'] - user_features['is_multiplayer']) < 0.3:
            if features['is_multiplayer'] > 0.5:
                reasons.append("modo multiplayer")
        
        if abs(features['is_singleplayer'] - user_features['is_singleplayer']) < 0.3:
            if features['is_singleplayer'] > 0.5:
                reasons.append("modo singleplayer")
        
        if abs(features['is_rpg'] - user_features['is_rpg']) < 0.3:
            if features['is_rpg'] > 0.5:
                reasons.append("gênero RPG")
        
        if abs(features['is_action'] - user_features['is_action']) < 0.3:
            if features['is_action'] > 0.5:
                reasons.append("gênero Action")
        
        if abs(features['is_strategy'] - user_features['is_strategy']) < 0.3:
            if features['is_strategy'] > 0.5:
                reasons.append("gênero Strategy")
        
        if abs(features['is_indie'] - user_features['is_indie']) < 0.3:
            if features['is_indie'] > 0.5:
                reasons.append("jogo indie")
        
        # Novas características
        if abs(features['supports_controller'] - user_features['supports_controller']) < 0.3:
            if features['supports_controller'] > 0.5:
                reasons.append("suporte a controle")
        
        # Verifica a popularidade
        if features['total_reviews'] > 10000:
            reasons.append("muito popular")
        elif features['total_reviews'] > 1000:
            reasons.append("popular")
        
        # Verifica a data de lançamento
        if features['years_since_release'] < 1:
            reasons.append("lançamento recente")
        elif features['years_since_release'] > 5:
            reasons.append("clássico")
        
        if reasons:
            return f"Recomendado por ter características similares ao seu gosto: {', '.join(reasons)} (similaridade: {similarity:.2f})"
        else:
            return f"Recomendado com base no seu perfil de jogos (similaridade: {similarity:.2f})"
    
    def save_profile(self, filename: str = "steam_profile.json"):
        """Salva o perfil do usuário em arquivo JSON"""
        if self.user_profile:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            print(f"✓ Perfil salvo em {filename}")
    
    def print_user_profile_summary(self):
        """Imprime um resumo do perfil do usuário"""
        if not self.user_profile:
            print("Perfil do usuário não foi construído ainda.")
            return
        
        print("\n=== RESUMO DO PERFIL DO USUÁRIO ===")
        print(f"Steam ID: {self.user_profile['steam_id']}")
        print(f"Jogos analisados: {self.user_profile['total_games_analyzed']}")
        
        print("\nCaracterísticas médias do seu gosto:")
        features = self.user_profile['avg_features']
        print(f"  • Preço médio: R${features['price']:.2f}")
        print(f"  • Avaliação positiva: {features['positive_ratio']:.2%}")
        print(f"  • Popularidade média: {features['total_reviews']:.0f} avaliações")
        print(f"  • Prefere jogos recentes: {'Sim' if features['years_since_release'] < 3 else 'Não'}")
        print(f"  • Prefere controle: {'Sim' if features['supports_controller'] > 0.5 else 'Não'}")
        print(f"  • Prefere multiplayer: {'Sim' if features['is_multiplayer'] > 0.5 else 'Não'}")
        print(f"  • Prefere singleplayer: {'Sim' if features['is_singleplayer'] > 0.5 else 'Não'}")
        print(f"  • Gosta de RPG: {'Sim' if features['is_rpg'] > 0.5 else 'Não'}")
        print(f"  • Gosta de Action: {'Sim' if features['is_action'] > 0.5 else 'Não'}")
        print(f"  • Gosta de Strategy: {'Sim' if features['is_strategy'] > 0.5 else 'Não'}")
        print(f"  • Gosta de Indie: {'Sim' if features['is_indie'] > 0.5 else 'Não'}")
        
        print(f"\nTop 5 jogos mais jogados:")
        for i, game in enumerate(self.user_profile['top_games'][:5], 1):
            playtime_hours = game['playtime'] // 60
            playtime_minutes = game['playtime'] % 60
            print(f"  {i}. {game['name']} - {playtime_hours}h {playtime_minutes}min")

# Interface Web Simples
app = Flask(__name__)

# Variáveis globais para armazenar o sistema e recomendações
recommender = None
recommendations_data = None

@app.route('/')
def index():
    """Página inicial com o perfil do usuário e recomendações"""
    global recommender, recommendations_data
    
    if recommender is None:
        return "Sistema não inicializado. Execute o script principal primeiro."
    
    return render_template('index.html', 
                          profile=recommender.user_profile,
                          recommendations=recommendations_data)

@app.route('/api/recommendations')
def api_recommendations():
    """API para obter as recomendações em formato JSON"""
    global recommendations_data
    
    if recommendations_data is None:
        return jsonify({"error": "Recomendações não geradas"})
    
    return jsonify(recommendations_data)

def main():
    """Função principal para demonstrar o sistema de recomendação"""
    
    # Configurações (substitua pelos seus dados)
    API_KEY = "x"  # Sua chave da API
    STEAM_ID = "x"  # Seu Steam ID no formato 64 bits
    
    if API_KEY == "SUA_CHAVE_DA_API_STEAM" or STEAM_ID == "SEU_STEAM_ID":
        print("ERRO: Configure sua chave da API e Steam ID primeiro!")
        print("1. Obtenha sua chave da API em: https://steamcommunity.com/dev/apikey")
        print("2. Encontre seu Steam ID em: https://steamid.io/")
        print("3. Edite as variáveis API_KEY e STEAM_ID no código.")
        return
    
    global recommender, recommendations_data
    
    try:
        # Inicializa o sistema de recomendação
        print("=== SISTEMA DE RECOMENDAÇÃO HÍBRIDO - STEAM ===")
        print("Baseado no TCC: Sistema de Recomendação para Músicas, Filmes e Jogos")
        print()
        
        recommender = SteamRecommendationSystem(API_KEY, STEAM_ID)
        
        # Constrói o perfil do usuário (forçando reconstrução para garantir dados atualizados)
        user_profile = recommender.build_user_profile(limit=50, force_rebuild=True)
        recommender.print_user_profile_summary()
        
        # Gera recomendações baseadas em diferentes gêneros
        search_queries = [
            "RPG",
            "Action",
            "Strategy",
            "Indie",
            "Simulation"
        ]
        
        print(f"\nGerando recomendações baseadas em: {', '.join(search_queries)}")
        recommendations_data = recommender.get_recommendations(
            search_queries, 
            num_recommendations=10,
            max_per_genre=2,  # Máximo 2 jogos por gênero para diversidade
            exclude_owned=True  # Exclui jogos já possuídos
        )
        
        # Exibe recomendações
        print("\n=== SUAS RECOMENDAÇÕES PERSONALIZADAS ===")
        for rec in recommendations_data:
            print(f"\n{rec['rank']}. {rec['game_info']['name']}")
            print(f"   Preço: R${rec['game_info']['price']:.2f}")
            print(f"   Gêneros: {', '.join(rec['game_info']['genres'])}")
            print(f"   {rec['recommendation_reason']}")
            print(f"   Steam: {rec['game_info']['url']}")
        
        print("\n=== SISTEMA DE RECOMENDAÇÃO EXECUTADO COM SUCESSO! ===")
        print("Este exemplo demonstra como implementar o sistema descrito no seu TCC.")
        print("\nIniciando interface web...")
        print("Acesse http://127.0.0.1:5000 no seu navegador para ver as recomendações.")
        
        # Inicia a interface web
        app.run(debug=False, port=5000)
        
    except Exception as e:
        print(f"Erro: {e}")
        print("\nVerifique se:")
        print("1. Sua chave da API da Steam está correta")
        print("2. Seu Steam ID está correto")
        print("3. Sua biblioteca de jogos está pública")
        print("4. Sua conexão com a internet está funcionando")

if __name__ == "__main__":
    main()