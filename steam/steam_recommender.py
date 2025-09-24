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
        self.user_name = None
        
        # Cache para requisições à API
        self.app_details_cache = {}
        
        # Lista expandida de gêneros para rastreamento
        self.all_genres = [
            'Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing',
            'Indie', 'Casual', 'Massively Multiplayer', 'Free to Play', 'Early Access',
            'Animation & Modeling', 'Audio Production', 'Video Production', 'Design & Illustration',
            'Utilities', 'Web Publishing', 'Education', 'Software Training', 'Game Development',
            'Documentary', 'Accounting', 'Violent', 'Gore', 'Sexual Content', 'Nudity',
            'Atmospheric', 'Dark', 'Exploration', 'Puzzle', 'Platformer', 'Shooter',
            'Fighting', 'Horror', 'Survival', 'Stealth', 'Fantasy', 'Sci-Fi', 'Retro'
        ]
        
        # Lista de gêneros principais para preferências
        self.main_genres = [
            'Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing', 'Indie'
        ]
        
        # Lista de gêneros adicionais para preferências
        self.additional_genres = [
            'Casual', 'Massively Multiplayer', 'Free to Play', 'Early Access',
            'Puzzle', 'Platformer', 'Shooter', 'Horror'
        ]
        
    def get_user_summary(self) -> Dict:
        """Obtém informações básicas do usuário"""
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        params = {
            'key': self.api_key,
            'steamids': self.steam_id
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            players = data.get('response', {}).get('players', [])
            
            if players:
                return players[0]
            return {}
            
        except Exception as e:
            print(f"Erro ao obter informações do usuário: {e}")
            return {}
    
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
        
        # Obtém informações do usuário
        user_info = self.get_user_summary()
        self.user_name = user_info.get('personaname', 'Usuário Steam')
        
        # Obtém jogos do usuário
        owned_games = self.get_owned_games()
        
        if not owned_games:
            raise ValueError("Usuário não possui jogos na biblioteca")
        
        # Armazena IDs dos jogos possuídos
        self.owned_games_ids = {game['appid'] for game in owned_games}
        
        # Ordena por tempo de jogo (prioridade para jogos mais jogados)
        sorted_games = sorted(owned_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
        
        # Limita número de jogos
        games_to_analyze = sorted_games[:limit]
        
        # Extrai características dos jogos
        feature_vectors = []
        analyzed_games = []
        paid_games_prices = []  # Para calcular média de preços sem jogos grátis
        
        for game in games_to_analyze:
            app_id = game['appid']
            details = self.get_app_details(app_id)
            
            if not details:
                continue
                
            # Extrai características relevantes
            genres = [g['description'] for g in details.get('genres', [])]
            categories = [c['description'] for c in details.get('categories', [])]
            
            # Características numéricas
            price_info = details.get('price_overview', {})
            price = price_info.get('final', 0) / 100.0
            
            # Adiciona à lista de preços pagos se não for gratuito
            if price > 0:
                paid_games_prices.append(price)
            
            positive = details.get('positive', 0)
            total = details.get('total', 1)
            positive_ratio = positive / total if total > 0 else 0
            
            release_date = details.get('release_date', {}).get('date', '')
            years_since_release = 0
            if release_date:
                try:
                    year = int(release_date.split(',')[-1].strip())
                    current_year = datetime.datetime.now().year
                    years_since_release = current_year - year
                except:
                    pass
            
            supports_controller = 1 if 'Full Controller Support' in categories else 0
            
            # Cria dicionário de características de gênero
            genre_features = {}
            for genre in self.all_genres:
                genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
                genre_features[genre_key] = 1 if genre in genres else 0
            
            # Características principais
            features = {
                'price': price,
                'positive_ratio': positive_ratio,
                'total_reviews': total,
                'years_since_release': years_since_release,
                'supports_controller': supports_controller,
                'is_multiplayer': 1 if 'Multi-player' in categories else 0,
                'is_singleplayer': 1 if 'Single-player' in categories else 0,
                'is_coop': 1 if 'Co-op' in categories else 0,
                'playtime': game.get('playtime_forever', 0)
            }
            
            # Adiciona características de gênero
            features.update(genre_features)
            
            # Garante que todas as características estejam presentes
            all_feature_keys = [
                'price', 'positive_ratio', 'total_reviews', 'years_since_release',
                'supports_controller', 'is_multiplayer', 'is_singleplayer', 'is_coop', 'playtime'
            ]
            
            # Adiciona chaves de gênero
            for genre in self.all_genres:
                genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
                all_feature_keys.append(genre_key)
            
            # Garante que todas as chaves existam no dicionário features
            for key in all_feature_keys:
                if key not in features:
                    features[key] = 0
            
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
        
        # Normaliza as características contínuas
        continuous_indices = [0, 1, 2, 3, 8]  # price, positive_ratio, total_reviews, years_since_release, playtime (índice corrigido)
        self.scaler.fit(feature_matrix[:, continuous_indices])
        feature_matrix[:, continuous_indices] = self.scaler.transform(feature_matrix[:, continuous_indices])
        
        # Calcula características médias do usuário
        user_avg_features = np.mean(feature_matrix, axis=0)
        
        # Depuração: imprimir informações sobre o vetor do usuário
        print("Dimensão do vetor do usuário:", user_avg_features.shape)
        print("Primeiros 10 valores do vetor do usuário:", user_avg_features[:10])
        
        # Armazena o perfil do usuário
        self.user_features_vector = user_avg_features
        
        # Calcula média de preços apenas de jogos pagos
        avg_paid_price = np.mean(paid_games_prices) if paid_games_prices else 0
        
        # Cria perfil detalhado
        self.user_profile = {
            'steam_id': self.steam_id,
            'user_name': self.user_name,
            'total_games_analyzed': len(analyzed_games),
            'avg_paid_price': avg_paid_price,  # Nova: média de preços sem jogos grátis
            'avg_features': {}
        }
        
        # Adiciona características principais ao perfil
        main_features = ['price', 'positive_ratio', 'total_reviews', 'years_since_release', 
                        'supports_controller', 'is_multiplayer', 'is_singleplayer', 'is_coop', 'playtime']
        for i, feature in enumerate(main_features):
            self.user_profile['avg_features'][feature] = float(user_avg_features[i])
        
        # Adiciona preferências de gêneros principais
        for genre in self.main_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            # Verifica se o gênero está presente nas características
            if genre_key in features:
                idx = list(features.keys()).index(genre_key)
                self.user_profile['avg_features'][genre_key] = float(user_avg_features[idx])
            else:
                # Se não estiver, adiciona como 0
                self.user_profile['avg_features'][genre_key] = 0.0
        
        # Adiciona preferências de gêneros adicionais
        for genre in self.additional_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            # Verifica se o gênero está presente nas características
            if genre_key in features:
                idx = list(features.keys()).index(genre_key)
                self.user_profile['avg_features'][genre_key] = float(user_avg_features[idx])
            else:
                # Se não estiver, adiciona como 0
                self.user_profile['avg_features'][genre_key] = 0.0
        
        # Verifica se todas as características necessárias estão presentes
        required_features = [
            'price', 'positive_ratio', 'total_reviews', 'years_since_release',
            'supports_controller', 'is_multiplayer', 'is_singleplayer', 'is_coop', 'playtime'
        ]

        for feature in required_features:
            if feature not in self.user_profile['avg_features']:
                print(f"Característica ausente no perfil: {feature}")
                self.user_profile['avg_features'][feature] = 0.0

        # Verifica gêneros principais
        for genre in self.main_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key not in self.user_profile['avg_features']:
                print(f"Gênero principal ausente no perfil: {genre}")
                self.user_profile['avg_features'][genre_key] = 0.0

        # Verifica gêneros adicionais
        for genre in self.additional_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key not in self.user_profile['avg_features']:
                print(f"Gênero adicional ausente no perfil: {genre}")
                self.user_profile['avg_features'][genre_key] = 0.0
        
        self.user_profile['top_games'] = analyzed_games[:10]
        
        # Salva o perfil automaticamente
        self.save_profile()
        
        print(f"✓ Perfil construído com {len(analyzed_games)} jogos")
        print(f"✓ Média de preço (jogos pagos): R${avg_paid_price:.2f}")
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
                price = price_info.get('final', 0) / 100.0
                
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
                        year = int(release_date.split(',')[-1].strip())
                        current_year = datetime.datetime.now().year
                        years_since_release = current_year - year
                    except:
                        pass
                
                # Suporte a controle
                supports_controller = 1 if 'Full Controller Support' in categories else 0
                
                # Cria dicionário de características de gênero
                genre_features = {}
                for genre in self.all_genres:
                    genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
                    genre_features[genre_key] = 1 if genre in genres else 0
                
                # Características principais
                features = {
                    'price': price,
                    'positive_ratio': positive_ratio,
                    'total_reviews': total,
                    'years_since_release': years_since_release,
                    'supports_controller': supports_controller,
                    'is_multiplayer': 1 if 'Multi-player' in categories else 0,
                    'is_singleplayer': 1 if 'Single-player' in categories else 0,
                    'is_coop': 1 if 'Co-op' in categories else 0,
                    'playtime': 0
                }
                
                # Adiciona características de gênero
                features.update(genre_features)
                
                # Garante que todas as características estejam presentes
                all_feature_keys = [
                    'price', 'positive_ratio', 'total_reviews', 'years_since_release',
                    'supports_controller', 'is_multiplayer', 'is_singleplayer', 'is_coop', 'playtime'
                ]
                
                # Adiciona chaves de gênero
                for genre in self.all_genres:
                    genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
                    all_feature_keys.append(genre_key)
                
                # Garante que todas as chaves existam no dicionário features
                for key in all_feature_keys:
                    if key not in features:
                        features[key] = 0
                
                # Cria vetor de características normalizadas
                feature_vector = list(features.values())
                
                # Normaliza as características contínuas usando o scaler do perfil do usuário
                continuous_indices = [0, 1, 2, 3, 8]  # price, positive_ratio, total_reviews, years_since_release, playtime (índice corrigido)
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_array[:, continuous_indices] = self.scaler.transform(feature_array[:, continuous_indices])
                feature_vector = feature_array.flatten().tolist()
                
                # Depuração: imprimir informações sobre o vetor do jogo
                print(f"Dimensão do vetor do jogo {game['name']}: {np.array(feature_vector).shape}")
                print(f"Primeiros 10 valores do vetor do jogo {game['name']}: {feature_vector[:10]}")
                
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
        print("Calculando similaridades baseadas em conteúdo...")
        
        # Extrai vetores de características dos jogos
        game_vectors = []
        for game in games:
            if 'feature_vector' in game and len(game['feature_vector']) == len(self.user_features_vector):
                game_vectors.append(game['feature_vector'])
            else:
                print(f"Vetor de características inválido para o jogo: {game.get('name', 'Desconhecido')}")
        
        if not game_vectors:
            print("Nenhum vetor de características válido encontrado!")
            return []
        
        game_vectors = np.array(game_vectors)
        user_vector = self.user_features_vector.reshape(1, -1)
        
        # Verifica dimensões
        if user_vector.shape[1] != game_vectors.shape[1]:
            print(f"Dimensões incompatíveis: usuário {user_vector.shape}, jogos {game_vectors.shape}")
            return []
        
        # Calcula similaridade do cosseno
        similarities = cosine_similarity(user_vector, game_vectors)[0]
        
        # Combina jogos com suas similaridades
        game_similarities = list(zip(games, similarities))
        
        # Ordena por similaridade (maior primeiro)
        game_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✓ Similaridades calculadas para {len(games)} jogos")
        return game_similarities
    
    def get_recommendations(self, search_queries: List[str], num_recommendations: int = 10) -> List[Dict]:
        """
        Gera recomendações baseadas apenas no perfil geral do usuário
        
        Args:
            search_queries (List[str]): Lista de termos de busca
            num_recommendations (int): Número de recomendações a retornar
            
        Returns:
            List[Dict]: Lista de recomendações ordenadas por relevância
        """
        if self.user_profile is None:
            raise ValueError("Perfil do usuário não foi construído. Execute build_user_profile() primeiro.")
        
        print(f"Gerando {num_recommendations} recomendações baseadas no perfil geral...")
        
        # Busca jogos candidatos
        all_games = []
        for query in search_queries:
            games = self.search_games(query, limit=20)
            all_games.extend(games)
        
        # Remove duplicatas
        unique_games = {}
        for game in all_games:
            if game['id'] not in unique_games:
                unique_games[game['id']] = game
        
        unique_games_list = list(unique_games.values())
        print(f"✓ Total de jogos únicos encontrados: {len(unique_games_list)}")
        
        # Filtra jogos já possuídos
        filtered_games = [game for game in unique_games_list if game['id'] not in self.owned_games_ids]
        print(f"✓ Jogos após filtrar os já possuídos: {len(filtered_games)}")
        
        # Calcula similaridades
        game_similarities = self.calculate_content_based_similarity(filtered_games)
        
        # Prepara recomendações finais
        recommendations = []
        for game, similarity in game_similarities[:num_recommendations]:
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
        
        # Analisa preferências de gêneros principais
        for genre in self.main_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key in features and genre_key in user_features:
                if abs(features[genre_key] - user_features[genre_key]) < 0.3:
                    if features[genre_key] > 0.5:
                        reasons.append(f"gênero {genre}")
        
        # Analisa preferências de gêneros adicionais
        for genre in self.additional_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key in features and genre_key in user_features:
                if abs(features[genre_key] - user_features[genre_key]) < 0.3:
                    if features[genre_key] > 0.5:
                        reasons.append(f"gênero {genre}")
        
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
        
        # Classificação da similaridade
        if similarity >= 0.8:
            similarity_class = "Excelente combinação"
            explanation = "Este jogo combina perfeitamente com seu perfil!"
        elif similarity >= 0.6:
            similarity_class = "Boa combinação"
            explanation = "Este jogo combina bem com suas preferências."
        elif similarity >= 0.4:
            similarity_class = "Combinação moderada"
            explanation = "Este jogo tem algumas características que você gosta."
        else:
            similarity_class = "Combinação fraca"
            explanation = "Este jogo é diferente do seu perfil, mas pode ser interessante para explorar novos gêneros."
        
        if reasons:
            return f"{explanation} Características em comum: {', '.join(reasons)}. (similaridade: {similarity:.2f} - {similarity_class})"
        else:
            return f"{explanation} (similaridade: {similarity:.2f} - {similarity_class})"
    
    def save_profile(self, filename: str = "steam_profile.json"):
        """Salva o perfil do usuário em arquivo JSON"""
        if self.user_profile:
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            print(f"✓ Perfil salvo em {filename}")
    
    def print_user_profile_summary(self):
        """Imprime um resumo do perfil do usuário"""
        if not self.user_profile:
            print("Perfil do usuário não foi construído ainda.")
            return
        
        print("\n=== RESUMO DO PERFIL DO USUÁRIO ===")
        print(f"Nome: {self.user_profile.get('user_name', 'Usuário Steam')}")
        print(f"Jogos analisados: {self.user_profile['total_games_analyzed']}")
        
        print("\nCaracterísticas médias do seu gosto:")
        features = self.user_profile['avg_features']
        print(f"  • Preço médio: R${features['price']:.2f}")
        print(f"  • Preço médio (jogos pagos): R${self.user_profile.get('avg_paid_price', 0):.2f}")
        print(f"  • Avaliação positiva: {features['positive_ratio']:.2%}")
        print(f"  • Popularidade média: {features['total_reviews']:.0f} avaliações")
        print(f"  • Prefere jogos recentes: {'Sim' if features['years_since_release'] < 3 else 'Não'}")
        print(f"  • Prefere controle: {'Sim' if features['supports_controller'] > 0.5 else 'Não'}")
        print(f"  • Prefere multiplayer: {'Sim' if features['is_multiplayer'] > 0.5 else 'Não'}")
        print(f"  • Prefere singleplayer: {'Sim' if features['is_singleplayer'] > 0.5 else 'Não'}")
        
        print("\nPreferências de Gêneros Principais:")
        for genre in self.main_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key in features:
                print(f"  • {genre}: {'Sim' if features[genre_key] > 0.5 else 'Não'}")
        
        print("\nPreferências de Gêneros Adicionais:")
        for genre in self.additional_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key in features:
                print(f"  • {genre}: {'Sim' if features[genre_key] > 0.5 else 'Não'}")
        
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
                          recommendations=recommendations_data,
                          main_genres=recommender.main_genres,
                          additional_genres=recommender.additional_genres)

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
    API_KEY = "07D56861E698CE5D1022898E730D62A4"  # Sua chave da API
    STEAM_ID = "76561198881909909"  # Seu Steam ID no formato 64 bits
    
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
        
        # Constrói o perfil do usuário
        user_profile = recommender.build_user_profile(limit=50, force_rebuild=True)
        recommender.print_user_profile_summary()
        
        # Gera recomendações baseadas no perfil geral
        search_queries = [
            "RPG",
            "Action",
            "Strategy",
            "Indie",
            "Simulation"
        ]
        
        print(f"\nGerando recomendações baseadas em: {', '.join(search_queries)}")
        recommendations_data = recommender.get_recommendations(search_queries, num_recommendations=10)
        
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
