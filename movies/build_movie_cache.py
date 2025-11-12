# build_movie_cache.py
# VERSÃO 5.4 - A VERDADE SIMPLES: USA .split(',')

import pandas as pd
import json
import pickle
import sys
from collections import Counter

# Não precisamos mais de 'ast'
# import ast

try:
    import pyarrow
except ImportError:
    print("\nERRO: A biblioteca 'pyarrow' não está instalada. Use: pip install pyarrow\n")
    sys.exit(1)

from sklearn.feature_extraction.text import TfidfVectorizer

MIN_VOTE_COUNT = 50
GENRE_BLACKLIST = ['Erotic', 'TV Movie']

print("--- INICIANDO CONSTRUÇÃO DO CACHE (v5.4 - A Verdade) ---")
try:
    print("\n[PASSO 1/9] Carregando dataset original...")
    df = pd.read_csv('TMDB_movie_dataset.csv', low_memory=False)
    print(f"-> OK. {len(df)} linhas carregadas.")

    print("\n[PASSO 2/9] Filtrando conteúdo adulto...")
    df['adult'] = df['adult'].astype(str).str.lower().fillna('false') == 'true'
    df = df[df['adult'] == False]
    print(f"-> OK. {len(df)} filmes restantes.")

    print(f"\n[PASSO 3/9] Filtrando por contagem de votos (mínimo: {MIN_VOTE_COUNT})...")
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    df = df[df['vote_count'] >= MIN_VOTE_COUNT]
    print(f"-> OK. {len(df)} filmes restantes.")

    # --- AQUI ESTÁ A CORREÇÃO HUMILHANTE E DEFINITIVA ---
    print("\n[PASSO 4/9] Processando a coluna 'genres' como texto simples (split)...")
    def get_genres_from_string(data_str):
        try:
            # Apenas separa a string pela vírgula e remove espaços em branco
            return [genre.strip() for genre in str(data_str).split(',') if genre.strip()]
        except:
            return []
            
    df['genres_list'] = df['genres'].apply(get_genres_from_string)
    empty_genres_count = len(df[df['genres_list'].apply(len) == 0])
    print(f"-> OK. {len(df)} filmes processados. {empty_genres_count} filmes ficaram sem gênero.")
    # --- FIM DA CORREÇÃO ---

    print(f"\n[PASSO 5/9] Removendo filmes com gêneros da blacklist ({GENRE_BLACKLIST})...")
    df = df[df['genres_list'].apply(lambda g: not any(blacklisted in genre for blacklisted in GENRE_BLACKLIST for genre in g))]
    print(f"-> OK. {len(df)} filmes restantes.")

    print("\n[PASSO 6/9] Removendo linhas com dados essenciais faltando...")
    required_cols = ['id', 'title', 'overview', 'genres', 'keywords', 'release_date', 'popularity', 'vote_average', 'poster_path']
    df = df.dropna(subset=required_cols).copy()
    print(f"-> OK. {len(df)} filmes restantes após o dropna.")

    print("\n[PASSO 7/9] Contando e salvando gêneros únicos...")
    all_genres = Counter([genre for sublist in df['genres_list'] for genre in sublist])
    if not all_genres:
        print("\n!!!!!!!!!! ALERTA CRÍTICO: NENHUM GÊNERO ENCONTRADO APÓS TODOS OS FILTROS !!!!!!!!!\n")
        sys.exit("Abortando devido à falha na coleta de gêneros.")
        
    clean_genres = sorted([genre for genre, count in all_genres.items() if count > 100])
    with open('genres.json', 'w', encoding='utf-8') as f:
        json.dump(clean_genres, f, ensure_ascii=False, indent=4)
    print(f"-> OK. Lista de {len(clean_genres)} gêneros salva em 'genres.json'.")

    print("\n[PASSO 8/9] Criando features de texto e matriz de similaridade...")
    df['genres_text'] = df['genres_list'].apply(lambda g: ' '.join([name.replace(' ', '').lower() for name in g]))
    # Keywords também são strings simples
    df['keywords_text'] = df['keywords'].fillna('').str.replace(',', ' ').str.lower()
    
    df['content_features'] = (df['overview'].fillna('') + ' ' + (df['genres_text'] + ' ') * 2 + (df['keywords_text'] + ' ') * 3)
    df['search_features'] = df['title'].str.lower() + ' ' + df['original_title'].str.lower().fillna('')
    df['id'] = pd.to_numeric(df['id'], errors='coerce').dropna().astype(int)
    df = df.dropna(subset=['id'])
    
    df_processed = df[['id', 'title', 'release_date', 'popularity', 'vote_average', 'genres', 'poster_path', 'search_features', 'content_features']].reset_index(drop=True)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_processed['content_features'])
    with open('movie_tfidf_matrix.pkl', 'wb') as f: pickle.dump(tfidf_matrix, f)
    print("-> OK. Matriz TF-IDF salva.")

    print("\n[PASSO 9/9] Salvando DataFrame processado no cache final...")
    df_to_save = df_processed.drop(columns=['content_features'])
    df_to_save.to_parquet('movies_processed.parquet', engine='pyarrow')
    print("-> OK. Cache de filmes salvo.")
    
    print(f"\n--- CONSTRUÇÃO DO CACHE (v5.4) CONCLUÍDA! {len(df_to_save)} filmes processados. ---")

except Exception as e:
    print(f"\n--- ERRO DURANTE A CONSTRUÇÃO DO CACHE ---"); import traceback; traceback.print_exc()
