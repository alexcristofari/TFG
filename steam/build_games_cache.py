# steam/build_games_cache.py (v1.7.1 - Correção do fillna)
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json

# --- Configurações ---
CACHE_DIR = 'cache'
INPUT_FILE = os.path.join(CACHE_DIR, 'games_processed_df.parquet')
# Arquivos de saída
GENRES_MATRIX_FILE = os.path.join(CACHE_DIR, 'genres_matrix.pkl')
CATEGORIES_MATRIX_FILE = os.path.join(CACHE_DIR, 'categories_matrix.pkl')
DESCRIPTION_MATRIX_FILE = os.path.join(CACHE_DIR, 'description_matrix.pkl')
DEVELOPERS_MATRIX_FILE = os.path.join(CACHE_DIR, 'developers_matrix.pkl')
VECTORIZER_FILE = os.path.join(CACHE_DIR, 'unified_vectorizer.pkl')

# --- INÍCIO DO PROCESSO ---
print("--- INICIANDO CONSTRUÇÃO DE CACHE VETORIAL (v1.7.1) ---")

# Garante que o diretório de cache exista
os.makedirs(CACHE_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    print(f"ERRO CRÍTICO: O arquivo de entrada '{INPUT_FILE}' não foi encontrado.")
    print("Certifique-se de que o arquivo de dados enriquecido da v1.6 está na pasta 'cache'.")
else:
    # PASSO 1: Carregar o DataFrame enriquecido
    print(f"[PASSO 1/4] Carregando dados de '{INPUT_FILE}'...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"-> OK. {len(df)} jogos carregados.")

    # PASSO 2: Preparar os campos de texto
    print("[PASSO 2/4] Preparando campos de texto para vetorização...")
    
    # --- CORREÇÃO AQUI ---
    # Substitui valores nulos em colunas de lista com uma lista vazia
    # A função apply garante que cada célula seja tratada individualmente
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    df['categories'] = df['categories'].apply(lambda x: x if isinstance(x, list) else [])
    df['developers'] = df['developers'].apply(lambda x: x if isinstance(x, list) else [])
    # Para strings, fillna com '' funciona bem
    df['short_description'].fillna('', inplace=True)
    # --- FIM DA CORREÇÃO ---

    df['genres_text'] = df['genres'].apply(lambda x: ' '.join(x))
    df['categories_text'] = df['categories'].apply(lambda x: ' '.join([cat['description'] for cat in x if 'description' in cat]))
    df['developers_text'] = df['developers'].apply(lambda x: ''.join(d.replace(' ', '') for d in x)) # Junta nomes de devs para tratá-los como uma única palavra
    df['description_text'] = df['short_description']
    print("-> OK. Campos de texto preparados.")

    # PASSO 3: Criar um vocabulário unificado
    print("[PASSO 3/4] Criando vocabulário unificado e vetorizando campos...")
    unified_corpus = pd.concat([
        df['genres_text'],
        df['categories_text'],
        df['description_text'],
        df['developers_text']
    ], ignore_index=True)

    unified_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    unified_vectorizer.fit(unified_corpus)
    
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(unified_vectorizer, f)
    print("-> OK. Vetorizador unificado salvo.")

    # PASSO 4: Vetorizar cada campo separadamente
    print("[PASSO 4/4] Vetorizando campos individuais e salvando matrizes...")
    genres_matrix = unified_vectorizer.transform(df['genres_text'])
    categories_matrix = unified_vectorizer.transform(df['categories_text'])
    description_matrix = unified_vectorizer.transform(df['description_text'])
    developers_matrix = unified_vectorizer.transform(df['developers_text'])

    with open(GENRES_MATRIX_FILE, 'wb') as f: pickle.dump(genres_matrix, f)
    with open(CATEGORIES_MATRIX_FILE, 'wb') as f: pickle.dump(categories_matrix, f)
    with open(DESCRIPTION_MATRIX_FILE, 'wb') as f: pickle.dump(description_matrix, f)
    with open(DEVELOPERS_MATRIX_FILE, 'wb') as f: pickle.dump(developers_matrix, f)
    print("-> OK. Matrizes de vetores individuais salvas.")

    # PASSO 5: EXTRAIR E SALVAR LISTA DE GÊNEROS PARA O FRONTEND
    print("[PASSO 5/5] Extraindo e salvando lista de gêneros...")
    all_genres = sorted(list(set(genre for sublist in df['genres'] for genre in sublist)))
    GAMES_GENRES_FILE = os.path.join(CACHE_DIR, 'games_genres.json')
    with open(GAMES_GENRES_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_genres, f, ensure_ascii=False, indent=4)
    print(f"-> OK. Lista de gêneros salva em '{GAMES_GENRES_FILE}'.")

    print("\n--- CONSTRUÇÃO DO CACHE VETORIAL CONCLUÍDA! ---")