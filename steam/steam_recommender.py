"""
Sistema de Recomendação Híbrido para Jogos - Versão Aprimorada e Otimizada
Este módulo implementa um sistema de recomendação que combina filtragem colaborativa
e filtragem baseada em conteúdo usando dados da Steam.
"""

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
from flask import Flask, render_template, jsonify, session
import logging
from logging.handlers import RotatingFileHandler
from requests.exceptions import HTTPError, Timeout, RequestException

# Importar biblioteca para gerenciar variáveis de ambiente
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env com tratamento de erro
try:
    load_dotenv()
except UnicodeDecodeError:
    print("⚠ Erro ao ler arquivo .env. Verifique a codificação do arquivo.")
    print("   O arquivo deve ser salvo com codificação UTF-8.")
    print("   Tente criar o arquivo .env novamente usando o Bloco de Notas com codificação UTF-8.")
    exit(1)

# Configurar logging sem emojis para evitar problemas de codificação no Windows
def setup_logging():
    """Configura sistema de logging robusto"""
    # Criar logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Formatter sem emojis
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler para arquivo (com rotação)
    file_handler = RotatingFileHandler('steam_recommender.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para console (sem emojis)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class SteamRecommendationSystem:
    """
    Sistema de Recomendação Híbrido para Jogos usando API da Steam
    
    Implementa os conceitos descritos no TCC:
    - Filtragem baseada em conteúdo usando características dos jogos
    - Perfil do usuário baseado em biblioteca de jogos
    - Normalização de características usando Min-Max
    - Similaridade do cosseno para recomendações
    """
    
    def __init__(self, api_key: str, steam_id: str, mode: str = "balanced"):
        """
        Inicializa o sistema de recomendação
        
        Args:
            api_key (str): Chave da API da Steam
            steam_id (str): ID do usuário na Steam (formato 64 bits)
            mode (str): Modo de operação
                - "fast": Rápido, usa cache sempre que possível
                - "balanced": Balanceado (padrão) - bom meio termo
                - "thorough": Completo, analisa tudo (pode bater rate limit)
        """
        self.api_key = api_key
        self.steam_id = steam_id
        self.base_url = "https://api.steampowered.com"
        self.mode = mode
        
        logger.info(f"Inicializando sistema para Steam ID: {steam_id} no modo {mode}")
        
        # Configurações por modo
        self.config = self._get_mode_config(mode)
        
        # Scaler para normalização das características
        self.scaler = MinMaxScaler()
        
        # Cache para armazenar dados
        self.user_profile = None
        self.user_features_vector = None
        self.games_database = []
        self.owned_games_ids = set()
        self.user_name = None
        
        # Cache persistente para requisições à API
        self.app_details_cache = {}
        self.cache_file = "steam_games_cache.json"
        self.cache_metadata = {}
        self._load_cache()
        
        # Contador de requisições
        self.request_count = 0
        self.cache_hits = 0
        
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
    
    def _get_mode_config(self, mode: str) -> Dict:
        """Retorna configurações baseadas no modo escolhido"""
        configs = {
            "fast": {
                "profile_limit": 25,
                "search_limit": 30,
                "max_queries": 5,
                "sleep_time": 0.3,
                "pause_every": 15,
                "pause_duration": 2,
                "prefer_cache": True
            },
            "balanced": {
                "profile_limit": 40,
                "search_limit": 60,
                "max_queries": 8,
                "sleep_time": 0.35,
                "pause_every": 12,
                "pause_duration": 3,
                "prefer_cache": False
            },
            "thorough": {
                "profile_limit": 60,
                "search_limit": 100,
                "max_queries": 12,
                "sleep_time": 0.5,
                "pause_every": 8,
                "pause_duration": 4,
                "prefer_cache": False
            }
        }
        return configs.get(mode, configs["balanced"])
    
    def _load_cache(self):
        """Carrega cache com informação de idade"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Se o cache tem metadata
                if "metadata" in cache_data:
                    self.cache_metadata = cache_data["metadata"]
                    self.app_details_cache = cache_data["games"]
                else:
                    # Cache antigo sem metadata
                    self.app_details_cache = cache_data
                    self.cache_metadata = {"created": time.time()}
                    
                logger.info(f"Cache carregado: {len(self.app_details_cache)} jogos")
                
                # Mostra idade do cache
                age_days = (time.time() - self.cache_metadata.get("created", time.time())) / 86400
                if age_days > 30:
                    logger.warning(f"Cache tem {age_days:.0f} dias. Considere rebuild para dados atualizados.")
                
            except Exception as e:
                logger.error(f"Erro ao carregar cache: {e}")
                self.app_details_cache = {}
                self.cache_metadata = {"created": time.time()}
        else:
            logger.info("Criando novo cache...")
            self.app_details_cache = {}
            self.cache_metadata = {"created": time.time()}
    
    def _save_cache_batch(self):
        """Salva cache com metadata"""
        try:
            cache_data = {
                "metadata": {
                    "created": self.cache_metadata.get("created", time.time()),
                    "updated": time.time(),
                    "total_games": len(self.app_details_cache),
                    "mode": self.mode
                },
                "games": self.app_details_cache
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
        
    def get_friends_list(self) -> List[Dict]:
        """Obtém a lista de amigos do usuário"""
        url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
        params = {
            'key': self.api_key,
            'steamid': self.steam_id,
            'relationship': 'friend'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            friends = data.get('friendslist', {}).get('friends', [])
            logger.info(f"Obtidos {len(friends)} amigos")
            return friends

        except HTTPError as http_err:
            if response.status_code == 401:
                logger.error("Chave da API inválida")
            elif response.status_code == 403:
                logger.error("Acesso negado à lista de amigos")
            else:
                logger.error(f"Erro HTTP ao obter amigos: {http_err}")
        except Exception as e:
            logger.error(f"Erro ao obter lista de amigos: {e}")
        return []

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
                logger.info(f"Usuário encontrado: {players[0].get('personaname', 'Unknown')}")
                return players[0]
            return {}
            
        except HTTPError as http_err:
            if response.status_code == 401:
                logger.error("Chave da API inválida")
            else:
                logger.error(f"Erro HTTP ao obter usuário: {http_err}")
        except Exception as e:
            logger.error(f"Erro ao obter informações do usuário: {e}")
        return {}
    
    def get_owned_games_for_id(self, steam_id: str) -> List[Dict]:
        """Obtém a biblioteca de jogos do usuário"""
        url = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/"
        params = {
            'key': self.api_key,
            'steamid': steam_id,
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
            logger.debug(f"Obtidos {len(games)} jogos para {steam_id}")
            return games
            
        except HTTPError as http_err:
            if response.status_code == 401:
                # Não levanta erro para amigos, apenas retorna vazio
                return []
            elif response.status_code == 403:
                # Acesso negado, biblioteca privada, retorna vazio
                return []
            elif response.status_code == 400:
                logger.error("Requisição inválida. Verifique o formato do Steam ID")
                raise ValueError("Requisição inválida. Verifique o formato do Steam ID")
            else:
                logger.error(f"Erro HTTP ao obter jogos para {steam_id}: {http_err}")
                return []
        except Exception as err:
            logger.error(f"Erro ao obter jogos para {steam_id}: {err}")
            return []

    def get_owned_games(self) -> List[Dict]:
        """Obtém a biblioteca de jogos do usuário"""
        return self.get_owned_games_for_id(self.steam_id)
    
    def get_app_details(self, app_id: int, max_retries: int = 3) -> Dict:
        """
        Obtém detalhes de um jogo específico com cache persistente e retry
        
        Args:
            app_id (int): ID do jogo
            max_retries (int): Número máximo de tentativas
        
        Returns:
            Dict: Detalhes do jogo ou dicionário vazio
        """
        # Verifica cache primeiro (converte para string para compatibilidade JSON)
        cache_key = str(app_id)
        if cache_key in self.app_details_cache:
            self.cache_hits += 1
            return self.app_details_cache[cache_key]
        
        # Nova requisição
        self.request_count += 1
        
        url = "https://store.steampowered.com/api/appdetails"
        params = {
            'appids': app_id,
            'cc': 'br'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                
                # Se bateu no rate limit (429), espera e tenta novamente
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    logger.warning(f"Rate limit! Aguardando {wait_time}s... (tentativa {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                data = response.json()
                if str(app_id) in data and data[str(app_id)]['success']:
                    details = data[str(app_id)]['data']
                    self.app_details_cache[cache_key] = details
                    return details
                
                return {}
                
            except Timeout:
                logger.warning(f"Timeout ao buscar {app_id}. Tentativa {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Erro ao obter detalhes do app {app_id}: {e}")
                    return {}
                time.sleep(2 ** attempt)
        
        return {}
    
    def build_user_profile(self, limit: int = None, force_rebuild: bool = False) -> Dict:
        """
        Constrói o perfil do usuário baseado em sua biblioteca de jogos
        
        Args:
            limit (int): Número de jogos a analisar (None usa configuração do modo)
            force_rebuild (bool): Se True, força a reconstrução do perfil mesmo se já existir
            
        Returns:
            Dict: Perfil do usuário com características médias
        """
        # Verifica se já temos um perfil salvo e se não for para forçar reconstrução
        if not force_rebuild and self.user_profile is not None:
            logger.info("Usando perfil existente em memória...")
            return self.user_profile
            
        # Tenta carregar do arquivo
        if not force_rebuild and os.path.exists("steam_profile.json"):
            try:
                with open("steam_profile.json", "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                    self.user_profile = profile_data
                    
                # Carrega o scaler salvo
                if os.path.exists("steam_scaler.pkl"):
                    import pickle
                    with open("steam_scaler.pkl", "rb") as f:
                        self.scaler = pickle.load(f)
                    logger.info("Scaler carregado do arquivo")
                else:
                    logger.warning("Scaler não encontrado, será necessário reconstruir o perfil")
                    force_rebuild = True
                    
                # Verifica se o perfil tem todas as características necessárias
                required_features = ['price', 'positive_ratio', 'total_reviews', 
                                   'years_since_release', 'supports_controller']
                for feature in required_features:
                    if feature not in self.user_profile.get('avg_features', {}):
                        logger.warning(f"Perfil salvo não contém a característica '{feature}'. Reconstruindo...")
                        force_rebuild = True
                        break
                
                if not force_rebuild:
                    # Reconstrói o vetor de features do usuário
                    self._rebuild_user_vector_from_profile()
                    logger.info("Perfil carregado do arquivo...")
                    return self.user_profile
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Erro ao carregar perfil salvo: {e}")
        
        # Usa limite do modo se não especificado
        if limit is None:
            limit = self.config["profile_limit"]
                
        logger.info(f"Construindo perfil do usuário (modo: {self.mode}, limite: {limit})...")
        
        # Obtém informações do usuário
        user_info = self.get_user_summary()
        self.user_name = user_info.get('personaname', 'Usuário Steam')
        
        # Obtém jogos do usuário
        owned_games = self.get_owned_games()
        
        if not owned_games:
            raise ValueError("Usuário não possui jogos na biblioteca")
        
        # Armazena IDs dos jogos possuídos
        self.owned_games_ids = {game['appid'] for game in owned_games}
        
        # Prioriza jogos com mais de 1h de jogo (remove testes/bundles não jogados)
        played_games = [g for g in owned_games if g.get('playtime_forever', 0) > 60]
        
        # Ordena por tempo de jogo (prioridade para jogos mais jogados)
        sorted_games = sorted(played_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
        
        # Limita número de jogos
        games_to_analyze = sorted_games[:limit]
        
        logger.info(f"Analisando top {len(games_to_analyze)} jogos (de {len(owned_games)} totais)")
        logger.info(f"Tempo total jogado: {sum(g.get('playtime_forever', 0) for g in games_to_analyze)//60:.0f}h")
        
        # Reset contadores
        self.request_count = 0
        self.cache_hits = 0
        
        # Extrai características dos jogos
        feature_vectors = []
        analyzed_games = []
        paid_games_prices = []  # Para calcular média de preços sem jogos grátis
        
        for i, game in enumerate(games_to_analyze):
            app_id = game['appid']
            
            # Progresso visual
            if (i + 1) % 5 == 0 or i == 0:
                cache_rate = (self.cache_hits / (self.cache_hits + self.request_count) * 100) if (self.cache_hits + self.request_count) > 0 else 0
                logger.info(f"[{i+1}/{len(games_to_analyze)}] Cache: {cache_rate:.0f}% | Novas requisições: {self.request_count}")
            
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
            
            # Pausa inteligente baseada no modo
            if (i + 1) % self.config["pause_every"] == 0:
                logger.info(f"Pausa estratégica de {self.config['pause_duration']}s...")
                time.sleep(self.config["pause_duration"])
                self._save_cache_batch()
            else:
                time.sleep(self.config["sleep_time"])
        
        # Salva cache final
        self._save_cache_batch()
        
        if not feature_vectors:
            raise ValueError("Não foi possível obter características dos jogos")
        
        # Converte para numpy array
        feature_matrix = np.array(feature_vectors)
        
        # Normaliza as características contínuas
        continuous_indices = [0, 1, 2, 3, 8]  # price, positive_ratio, total_reviews, years_since_release, playtime
        self.scaler.fit(feature_matrix[:, continuous_indices])
        feature_matrix[:, continuous_indices] = self.scaler.transform(feature_matrix[:, continuous_indices])
        
        # Calcula características médias do usuário
        user_avg_features = np.mean(feature_matrix, axis=0)
        
        # Depuração: imprimir informações sobre o vetor do usuário
        logger.debug(f"Dimensão do vetor do usuário: {user_avg_features.shape}")
        logger.debug(f"Primeiros 10 valores do vetor do usuário: {user_avg_features[:10]}")
        
        # Armazena o perfil do usuário
        self.user_features_vector = user_avg_features
        
        # Calcula média de preços apenas de jogos pagos
        avg_paid_price = np.mean(paid_games_prices) if paid_games_prices else 0
        
        # Cria perfil detalhado
        self.user_profile = {
            'steam_id': self.steam_id,
            'user_name': self.user_name,
            'total_games_analyzed': len(analyzed_games),
            'avg_paid_price': avg_paid_price,
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
                logger.warning(f"Característica ausente no perfil: {feature}")
                self.user_profile['avg_features'][feature] = 0.0

        # Verifica gêneros principais
        for genre in self.main_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key not in self.user_profile['avg_features']:
                logger.warning(f"Gênero principal ausente no perfil: {genre}")
                self.user_profile['avg_features'][genre_key] = 0.0

        # Verifica gêneros adicionais
        for genre in self.additional_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key not in self.user_profile['avg_features']:
                logger.warning(f"Gênero adicional ausente no perfil: {genre}")
                self.user_profile['avg_features'][genre_key] = 0.0
        
        self.user_profile['top_games'] = analyzed_games[:10]
        
        # Salva o perfil automaticamente
        self.save_profile()
        
        # Estatísticas finais
        total_ops = self.cache_hits + self.request_count
        if total_ops > 0:
            cache_rate = (self.cache_hits / total_ops) * 100
            logger.info(f"Perfil construído com {len(analyzed_games)} jogos")
            logger.info("Estatísticas:")
            logger.info(f"   • Cache hits: {self.cache_hits}")
            logger.info(f"   • Novas requisições: {self.request_count}")
            logger.info(f"   • Taxa de cache: {cache_rate:.1f}%")
            logger.info(f"   • Média de preço (jogos pagos): R${avg_paid_price:.2f}")
        
        return self.user_profile
    
    def search_games(self, query: str, limit: int = None, prioritize_cache: bool = None) -> List[Dict]:
        """
        Busca jogos na Steam e analisa suas características
        
        Args:
            query (str): Termo de busca
            limit (int): Número de resultados (None usa configuração do modo)
            prioritize_cache (bool): Se True, prioriza jogos em cache
            
        Returns:
            List[Dict]: Lista de jogos com características analisadas
        """
        if limit is None:
            limit = self.config["search_limit"]
        
        if prioritize_cache is None:
            prioritize_cache = self.config["prefer_cache"]
        
        logger.info(f"Buscando jogos para: '{query}' (limite: {limit})...")
        
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
            games = data.get('items', [])
            
            # ESTRATÉGIA INTELIGENTE: Separa jogos em cache vs não-cache
            cached_games = []
            non_cached_games = []
            
            for game in games[:limit * 2]:  # Busca mais para filtrar
                if str(game['id']) in self.app_details_cache:
                    cached_games.append(game)
                else:
                    non_cached_games.append(game)
            
            # Prioriza cache se configurado
            if prioritize_cache:
                ordered_games = cached_games[:limit] + non_cached_games[:max(0, limit - len(cached_games))]
            else:
                # Mix balanceado
                ordered_games = []
                for i in range(limit):
                    if i < len(cached_games):
                        ordered_games.append(cached_games[i])
                    if i < len(non_cached_games) and len(ordered_games) < limit:
                        ordered_games.append(non_cached_games[i])
            
            logger.info(f"Jogos encontrados: {len(cached_games)} em cache, {len(non_cached_games)} novos")
            
            analyzed_games = []
            for i, game in enumerate(ordered_games[:limit]):
                app_id = game['id']
                
                was_cached = str(app_id) in self.app_details_cache
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
                continuous_indices = [0, 1, 2, 3, 8]  # price, positive_ratio, total_reviews, years_since_release, playtime
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_array[:, continuous_indices] = self.scaler.transform(feature_array[:, continuous_indices])
                feature_vector = feature_array.flatten().tolist()
                
                # Depuração: imprimir informações sobre o vetor do jogo
                logger.debug(f"Dimensão do vetor do jogo {game['name']}: {np.array(feature_vector).shape}")
                logger.debug(f"Primeiros 10 valores do vetor do jogo {game['name']}: {feature_vector[:10]}")
                
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
                
                # Pausa apenas se não estiver em cache
                if not was_cached:
                    if (i + 1) % self.config["pause_every"] == 0:
                        logger.info(f"Pausa de {self.config['pause_duration']}s... ({i + 1}/{len(ordered_games[:limit])})")
                        time.sleep(self.config["pause_duration"])
                    else:
                        time.sleep(self.config["sleep_time"])
            
            # Salva cache após busca
            self._save_cache_batch()
            
            cache_rate = (self.cache_hits / (self.cache_hits + self.request_count) * 100) if (self.cache_hits + self.request_count) > 0 else 0
            logger.info(f"Analisados {len(analyzed_games)} jogos (cache: {cache_rate:.0f}%)")
            return analyzed_games
            
        except Exception as e:
            logger.error(f"Erro na busca de jogos: {e}")
            return []
    
    def calculate_content_based_similarity(self, games: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Calcula similaridade baseada em conteúdo entre o perfil do usuário e os jogos
        
        Args:
            games (List[Dict]): Lista de jogos analisados
            
        Returns:
            List[Tuple[Dict, float]]: Lista de (jogo, similaridade) ordenada por similaridade
        """
        logger.info("Calculando similaridades baseadas em conteúdo...")
        
        # Extrai vetores de características dos jogos
        game_vectors = []
        for game in games:
            if 'feature_vector' in game and len(game['feature_vector']) == len(self.user_features_vector):
                game_vectors.append(game['feature_vector'])
            else:
                logger.warning(f"Vetor de características inválido para o jogo: {game.get('name', 'Desconhecido')}")
        
        if not game_vectors:
            logger.warning("Nenhum vetor de características válido encontrado!")
            return []
        
        game_vectors = np.array(game_vectors)
        user_vector = self.user_features_vector.reshape(1, -1)
        
        # Verifica dimensões
        if user_vector.shape[1] != game_vectors.shape[1]:
            logger.error(f"Dimensões incompatíveis: usuário {user_vector.shape}, jogos {game_vectors.shape}")
            return []
        
        # Calcula similaridade do cosseno
        # Adiciona uma pequena constante para evitar divisão por zero em vetores nulos
        user_vector_norm = np_linalg.norm(user_vector)
        game_vectors_norm = np_linalg.norm(game_vectors, axis=1)

        if user_vector_norm == 0 or np.any(game_vectors_norm == 0):
            logger.warning("Aviso: Vetor do usuário ou de algum jogo é nulo. Similaridade pode ser 0.")
            # Retorna similaridade 0 para vetores nulos
            similarities = np.zeros(game_vectors.shape[0])
        else:
            similarities = cosine_similarity(user_vector, game_vectors)[0]
        
        # Combina jogos com suas similaridades
        game_similarities = list(zip(games, similarities))
        
        # Ordena por similaridade (maior primeiro)
        game_similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Similaridades calculadas para {len(games)} jogos")
        return game_similarities
    
    def get_collaborative_recommendations(self, num_friends: int = 5, num_games_per_friend: int = 10) -> List[Dict]:
        """
        Gera recomendações baseadas em filtragem colaborativa, encontrando jogos que amigos semelhantes possuem
        e que o usuário principal ainda não tem.
        """
        logger.info("Gerando recomendações colaborativas...")
        similar_friends = self.get_similar_friends(num_friends=num_friends)
        if not similar_friends:
            logger.warning("Não foi possível encontrar amigos semelhantes para recomendações colaborativas.")
            return []

        friend_recommended_games = {}
        for friend_data, similarity in similar_friends:
            friend_steam_id = friend_data["steam_id"]
            friend_name = friend_data["name"]
            logger.info(f"Analisando jogos do amigo {friend_name} (similaridade: {similarity:.2f})...")
            
            owned_games_by_friend = self.get_owned_games_for_id(friend_steam_id)
            
            # Filtra jogos que o usuário principal já possui
            new_games_for_user = [game for game in owned_games_by_friend if game["appid"] not in self.owned_games_ids]
            
            # Ordena os jogos do amigo por tempo de jogo (mais jogados primeiro)
            sorted_friend_games = sorted(new_games_for_user, key=lambda x: x.get("playtime_forever", 0), reverse=True)
            
            for game in sorted_friend_games[:num_games_per_friend]:
                app_id = game["appid"]
                if app_id not in friend_recommended_games:
                    friend_recommended_games[app_id] = {
                        "game_info": {"name": game["name"], "appid": app_id},
                        "friends_who_own": [],
                        "total_playtime_by_friends": 0,
                        "avg_friend_similarity": 0
                    }
                friend_recommended_games[app_id]["friends_who_own"].append({"name": friend_name, "similarity": similarity})
                friend_recommended_games[app_id]["total_playtime_by_friends"] += game.get("playtime_forever", 0)
                
        # Calcula a média de similaridade dos amigos que possuem o jogo
        for app_id, data in friend_recommended_games.items():
            total_similarity = sum([f["similarity"] for f in data["friends_who_own"]])
            data["avg_friend_similarity"] = total_similarity / len(data["friends_who_own"])

        # Converte para lista e ordena por uma combinação de fatores (ex: avg_friend_similarity e total_playtime_by_friends)
        collaborative_recommendations = list(friend_recommended_games.values())
        collaborative_recommendations.sort(key=lambda x: (x["avg_friend_similarity"], x["total_playtime_by_friends"]), reverse=True)
        
        # Adiciona detalhes completos dos jogos recomendados
        final_collaborative_recs = []
        for rec in collaborative_recommendations:
            app_id = rec["game_info"]["appid"]
            details = self.get_app_details(app_id)
            if details:
                genres = [g["description"] for g in details.get("genres", [])]
                categories = [c["description"] for c in details.get("categories", [])]
                price_info = details.get("price_overview", {})
                price = price_info.get("final", 0) / 100.0
                
                rec["game_info"]["price"] = price
                rec["game_info"]["genres"] = genres
                rec["game_info"]["categories"] = categories
                rec["game_info"]["url"] = f"https://store.steampowered.com/app/{app_id}"
                rec["game_info"]["thumbnail"] = details.get("header_image", "")
                final_collaborative_recs.append(rec)
            time.sleep(0.1)

        logger.info(f"{len(final_collaborative_recs)} recomendações colaborativas geradas.")
        return final_collaborative_recs

    def calculate_weighted_score(self, similarity: float, total_reviews: int, positive_ratio: float) -> float:
        """
        Calcula uma pontuação ponderada combinando similaridade, número de avaliações e taxa de avaliações positivas.
        
        Args:
            similarity (float): Pontuação de similaridade baseada em conteúdo.
            total_reviews (int): Número total de avaliações do jogo.
            positive_ratio (float): Proporção de avaliações positivas do jogo (0 a 1).
            
        Returns:
            float: Pontuação ponderada final para o jogo.
        """
        # Pesos para cada componente (ajustáveis conforme a necessidade)
        weight_similarity = 0.6
        weight_reviews = 0.2  # Importância para jogos populares
        weight_positive = 0.2 # Importância para jogos bem avaliados

        # Normaliza o número de avaliações (log-scale para reduzir o impacto de valores extremos)
        # Adiciona 1 para evitar log(0)
        normalized_reviews = np.log1p(total_reviews) / np.log1p(10000) # Normaliza para um máximo de 10000 avaliações
        normalized_reviews = min(normalized_reviews, 1.0) # Garante que não exceda 1

        # Calcula a pontuação ponderada
        weighted_score = (
            weight_similarity * similarity +
            weight_reviews * normalized_reviews +
            weight_positive * positive_ratio
        )
        return weighted_score

    def get_recommendations(self, search_queries: List[str] = None, num_recommendations: int = 10, 
                            content_weight: float = 0.7, collaborative_weight: float = 0.3) -> List[Dict]:
        """
        Gera recomendações baseadas no perfil geral do usuário
        
        Args:
            search_queries (List[str]): Lista de termos de busca (None usa queries automáticas)
            num_recommendations (int): Número de recomendações a retornar
            
        Returns:
            List[Dict]: Lista de recomendações ordenadas por relevância
        """
        if self.user_profile is None:
            raise ValueError("Perfil do usuário não foi construído. Execute build_user_profile() primeiro.")
        
        # Se não forneceu queries, usa gêneros favoritos do usuário
        if search_queries is None:
            search_queries = self._get_smart_queries()
            logger.info(f"Queries automáticas baseadas no seu perfil: {search_queries}")
        
        # Limita queries baseado no modo
        max_queries = self.config["max_queries"]
        if len(search_queries) > max_queries:
            search_queries = search_queries[:max_queries]
            logger.info(f"Limitando a {max_queries} queries (modo {self.mode})")
        
        logger.info(f"Gerando {num_recommendations} recomendações (modo: {self.mode})...")
        
        # Reset stats
        self.request_count = 0
        self.cache_hits = 0
        
        # Busca jogos candidatos
        all_games = []
        for i, query in enumerate(search_queries):
            logger.info(f"[{i+1}/{len(search_queries)}] Buscando: '{query}'")
            games = self.search_games(query)
            all_games.extend(games)
            
            # Pausa entre queries
            if i < len(search_queries) - 1:
                time.sleep(1)
        
        # Remove duplicatas
        unique_games = {}
        for game in all_games:
            if game['id'] not in unique_games:
                unique_games[game['id']] = game
        
        unique_games_list = list(unique_games.values())
        logger.info(f"Total de jogos únicos encontrados: {len(unique_games_list)}")
        
        # Filtra jogos já possuídos
        filtered_games = [game for game in unique_games_list if game['id'] not in self.owned_games_ids]
        logger.info(f"Jogos após filtrar os já possuídos: {len(filtered_games)}")
        
        # Estatísticas da busca
        total_ops = self.cache_hits + self.request_count
        if total_ops > 0:
            cache_rate = (self.cache_hits / total_ops) * 100
            logger.info("Estatísticas da busca:")
            logger.info(f"   • Total de operações: {total_ops}")
            logger.info(f"   • Cache hits: {self.cache_hits} ({cache_rate:.1f}%)")
            logger.info(f"   • Novas requisições: {self.request_count}")
            logger.info(f"   • Jogos em cache agora: {len(self.app_details_cache)}")
        
        # Calcula similaridades
        game_similarities = self.calculate_content_based_similarity(filtered_games)
        
        # Prepara recomendações finais
        scored_recommendations = []
        for game, similarity in game_similarities:
            # Obtém total_reviews e positive_ratio para o jogo
            total_reviews = game['features']['total_reviews']
            positive_ratio = game['features']['positive_ratio']
            
            # Calcula a pontuação ponderada
            weighted_score = self.calculate_weighted_score(similarity, total_reviews, positive_ratio)
            
            scored_recommendations.append({
                'game': game,
                'similarity': similarity,
                'weighted_score': weighted_score
            })
            
        # Ordena as recomendações pela pontuação ponderada (maior primeiro)
        scored_recommendations.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        recommendations = []
        for i, rec_data in enumerate(scored_recommendations[:num_recommendations]):
            game = rec_data['game']
            similarity = rec_data['similarity']
            weighted_score = rec_data['weighted_score']
            
            recommendation = {
                'rank': i + 1,
                'similarity_score': float(similarity),
                'weighted_score': float(weighted_score),
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
                    'is_rpg': game["features"]["is_rpg"],
                    'is_action': game["features"]["is_action"],
                    'is_strategy': game["features"]["is_strategy"]
                },
                'recommendation_reason': self._generate_recommendation_reason(game, similarity, weighted_score)
            }
            recommendations.append(recommendation)
        
        logger.info(f"{len(recommendations)} recomendações geradas com sucesso!")
        return recommendations
    
    def _get_smart_queries(self) -> List[str]:
        """Gera queries inteligentes baseadas no perfil do usuário"""
        queries = []
        features = self.user_profile.get('avg_features', {})
        
        # Top 3 gêneros favoritos
        genre_scores = []
        for genre in self.main_genres + self.additional_genres:
            genre_key = f'is_{genre.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")}'
            if genre_key in features and features[genre_key] > 0.3:
                genre_scores.append((genre, features[genre_key]))
        
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        queries.extend([g[0] for g in genre_scores[:3]])
        
        # Adiciona queries complementares
        if features.get('is_multiplayer', 0) > 0.5:
            queries.append("Multiplayer")
        
        if features.get('positive_ratio', 0) > 0.8:
            queries.append("Highly Rated")
        
        # Garante pelo menos 3 queries
        if len(queries) < 3:
            queries.extend(["Popular", "Top Rated"])
        
        return queries[:self.config["max_queries"]]
    
    def _generate_recommendation_reason(self, game: Dict, similarity: float, weighted_score: float) -> str:
        """
        Gera uma explicação para a recomendação baseada nas características
        
        Args:
            game (Dict): Informações do jogo
            similarity (float): Score de similaridade
            weighted_score (float): Score ponderado
            
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
        
        # Classificação da recomendação com base na pontuação ponderada
        if weighted_score >= 0.8:
            recommendation_class = "Excelente recomendação"
            explanation = "Este jogo é uma excelente combinação com seu perfil, popularidade e avaliação!"
        elif weighted_score >= 0.6:
            recommendation_class = "Boa recomendação"
            explanation = "Este jogo é uma boa combinação com suas preferências, popularidade e avaliação."
        elif weighted_score >= 0.4:
            recommendation_class = "Recomendação moderada"
            explanation = "Este jogo tem algumas características que você gosta e uma boa avaliação/popularidade."
        else:
            recommendation_class = "Recomendação para explorar"
            explanation = "Este jogo pode ser interessante para explorar novos gêneros, com base em sua similaridade e popularidade."
        
        if reasons:
            return f"{explanation} Características em comum: {', '.join(reasons)}. (Pontuação: {weighted_score:.2f} - {recommendation_class})"
        else:
            return f"{explanation} (Pontuação: {weighted_score:.2f} - {recommendation_class})"
    
    def get_similar_friends(self, num_friends: int = 5) -> List[Tuple[Dict, float]]:
        """
        Encontra amigos com perfis de gosto semelhantes ao do usuário principal.
        """
        logger.info(f"Buscando {num_friends} amigos com perfis semelhantes...")
        friends_list = self.get_friends_list()
        if not friends_list:
            logger.warning("Nenhum amigo encontrado ou lista de amigos privada.")
            return []

        friend_profiles = []
        for friend in friends_list:
            friend_steam_id = friend["steamid"]
            # Evita construir perfil para o próprio usuário se ele estiver na lista de amigos (improvável, mas para segurança)
            if friend_steam_id == self.steam_id:
                continue
            
            # Obtém o nome do amigo
            friend_summary = self.get_user_summary_for_id(friend_steam_id)
            friend_name = friend_summary.get("personaname", f"Amigo {friend_steam_id}")

            profile = self.build_profile_for_id(friend_steam_id)
            if profile:
                friend_profiles.append({"steam_id": friend_steam_id, "name": friend_name, "profile": profile})
            time.sleep(0.1) # Respeitar limites da API

        if not friend_profiles:
            logger.warning("Não foi possível construir perfis para amigos.")
            return []

        user_vector = self.user_features_vector.reshape(1, -1)
        similar_friends = []

        for friend_data in friend_profiles:
            friend_vector = np.array(friend_data["profile"]["feature_vector"]).reshape(1, -1)
            
            # Garante que os vetores têm a mesma dimensão antes de calcular a similaridade
            if user_vector.shape[1] != friend_vector.shape[1]:
                logger.warning(f"Dimensões de vetor incompatíveis para o amigo {friend_data['name']}. Ignorando.")
                continue

            similarity = cosine_similarity(user_vector, friend_vector)[0][0]
            similar_friends.append((friend_data, similarity))

        similar_friends.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Encontrados {len(similar_friends)} amigos com perfis semelhantes.")
        return similar_friends[:num_friends]

    def get_user_summary_for_id(self, steam_id: str) -> Dict:
        """Obtém informações básicas de um usuário específico"""
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        params = {
            'key': self.api_key,
            'steamids': steam_id
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
            logger.error(f"Erro ao obter informações do usuário {steam_id}: {e}")
            return {}
    
    def build_profile_for_id(self, steam_id: str) -> Optional[Dict]:
        """Constrói perfil simplificado para um amigo (sem salvar)"""
        # Implementação simplificada que retorna None por padrão
        # Pode ser expandida para construir perfis de amigos se necessário
        return None

    def save_profile(self, filename: str = "steam_profile.json"):
        """Salva o perfil do usuário em arquivo JSON"""
        if self.user_profile:
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            logger.info(f"Perfil salvo em {filename}")
    
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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key-for-development")

# Variáveis globais para armazenar o sistema e recomendações
recommender = None
recommendations_data = None

@app.before_request
def init_recommender():
    """Inicializa o sistema de recomendação na sessão do usuário"""
    global recommender
    if recommender is None:
        try:
            recommender = SteamRecommendationSystem(
                api_key=os.getenv("STEAM_API_KEY"),
                steam_id=os.getenv("STEAM_ID"),
                mode="balanced"
            )
            logger.info("Sistema de recomendação inicializado")
        except Exception as e:
            logger.error(f"Erro ao inicializar recommender: {e}")
            recommender = None

@app.route('/')
def index():
    """Página inicial com o perfil do usuário e recomendações"""
    global recommender, recommendations_data
    
    if recommender is None:
        return "Sistema não inicializado. Verifique as variáveis de ambiente.", 500
    
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
        return jsonify({"error": "Recomendações não geradas"}), 404
    
    return jsonify(recommendations_data)

def main():
    """Função principal para demonstrar o sistema de recomendação"""
    
    # Obter configurações das variáveis de ambiente
    API_KEY = os.getenv("STEAM_API_KEY")
    STEAM_ID = os.getenv("STEAM_ID")
    
    # Verificar se as variáveis de ambiente foram definidas
    if not API_KEY or not STEAM_ID:
        logger.error("Configure as variáveis de ambiente STEAM_API_KEY e STEAM_ID!")
        print("ERRO: Configure as variáveis de ambiente STEAM_API_KEY e STEAM_ID!")
        print("1. Crie um arquivo .env na mesma pasta do script")
        print("2. Adicione as seguintes linhas ao arquivo .env:")
        print("   STEAM_API_KEY=sua_chave_aqui")
        print("   STEAM_ID=seu_steam_id_aqui")
        print("3. Substitua 'sua_chave_aqui' e 'seu_steam_id_aqui' pelos valores reais")
        return
    
    global recommender, recommendations_data
    
    try:
        # Inicializa o sistema de recomendação
        print("=== SISTEMA DE RECOMENDAÇÃO HÍBRIDO - STEAM ===")
        print("Baseado no TCC: Sistema de Recomendação para Músicas, Filmes e Jogos")
        print()
        
        # ESCOLHA DO MODO
        print("Escolha o modo de operação:")
        print("  1. FAST - Rápido, usa cache (melhor para testes)")
        print("  2. BALANCED - Balanceado (recomendado) ⭐")
        print("  3. THOROUGH - Completo, analisa tudo (pode ser lento)")
        
        mode_choice = input("\nEscolha (1/2/3) [padrão: 2]: ").strip() or "2"
        
        mode_map = {"1": "fast", "2": "balanced", "3": "thorough"}
        mode = mode_map.get(mode_choice, "balanced")
        
        print(f"\n✓ Modo selecionado: {mode.upper()}\n")
        
        recommender = SteamRecommendationSystem(API_KEY, STEAM_ID, mode=mode)
        
        # Constrói o perfil do usuário
        user_profile = recommender.build_user_profile(force_rebuild=False)
        recommender.print_user_profile_summary()
        
        # Gera recomendações baseadas no perfil geral
        # Pode usar queries automáticas ou especificar manualmente
        search_queries = None  # None = usa queries automáticas baseadas no perfil
        
        # Ou especifique manualmente:
        # search_queries = ["RPG", "Action", "Strategy", "Indie", "Simulation"]
        
        print(f"\nGerando recomendações...")
        recommendations_data = recommender.get_recommendations(search_queries, num_recommendations=10)
        
        # Salva recomendações em arquivo JSON para a web
        with open("recommendations.json", "w", encoding="utf-8") as f:
            json.dump(recommendations_data, f, indent=2, ensure_ascii=False)
        
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
        print(f"\nModo utilizado: {mode.upper()}")
        print(f"Jogos em cache: {len(recommender.app_details_cache)}")
        print("\nIniciando interface web...")
        print("Acesse http://127.0.0.1:5000 no seu navegador para ver as recomendações.")
        
        # Inicia a interface web
        app.run(debug=False, port=5000)
        
    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        print(f"Erro: {e}")
        print("\nVerifique se:")
        print("1. Suas variáveis de ambiente estão configuradas corretamente")
        print("2. Sua biblioteca de jogos está pública")
        print("3. Sua conexão com a internet está funcionando")

if __name__ == "__main__":
    main()
