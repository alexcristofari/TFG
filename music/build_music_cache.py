# music/build_music_cache.py (v1.0)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import os
import pickle
import json

print("--- INICIANDO CONSTRUÇÃO DE CACHE PARA MÚSICAS ---")

# --- Configurações ---
CACHE_DIR = 'cache'
INPUT_CSV = os.path.join(CACHE_DIR, 'spotify_dataset.csv')
OUTPUT_DF_PARQUET = os.path.join(CACHE_DIR, 'music_processed.parquet')
OUTPUT_MATRIX_PKL = os.path.join(CACHE_DIR, 'music_feature_matrix.pkl')
OUTPUT_GENRES_JSON = os.path.join(CACHE_DIR, 'music_genres.json')
OUTPUT_ENCODERS_PKL = os.path.join(CACHE_DIR, 'music_encoders.pkl') # Salvar os encoders é uma boa prática

# Garante que o diretório de cache exista
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Passo 1: Carregar e Limpar os Dados ---
print(f"[PASSO 1/3] Carregando e limpando dados de '{INPUT_CSV}'...")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"--- ERRO FATAL: Arquivo '{INPUT_CSV}' não encontrado. ---")
    print("Certifique-se de que o dataset do Spotify está na pasta 'music/cache/'.")
    exit()

df.columns = df.columns.str.strip()

audio_feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
required_cols = ['track_id', 'track_name', 'artists', 'track_genre', 'popularity'] + audio_feature_cols

df = df.dropna(subset=required_cols).copy()
df = df.rename(columns={'track_id': 'id', 'track_name': 'name', 'track_genre': 'genres'})
df = df.drop_duplicates(subset=['name', 'artists'])
df['artists'] = df['artists'].str.split(';').str[0]
df = df.reset_index(drop=True)
print(f"-> Dados limpos. {len(df)} faixas únicas.")

# --- Passo 2: Vetorizar Features ---
print("[PASSO 2/3] Vetorizando features (Gênero + Áudio)...")
genre_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
numerical_scaler = MinMaxScaler()

genre_matrix = genre_encoder.fit_transform(df[['genres']])
numerical_data = df[audio_feature_cols].astype(np.float32)
numerical_matrix = numerical_scaler.fit_transform(numerical_data)

W_GENRE, W_AUDIO = 0.6, 0.4
feature_matrix = hstack([
    genre_matrix.astype(np.float32) * W_GENRE,
    csr_matrix(numerical_matrix).astype(np.float32) * W_AUDIO
]).tocsr()
print("-> Matriz de features criada.")

# --- Passo 3: Salvar Artefatos em Cache ---
print("[PASSO 3/3] Salvando arquivos de cache...")
df.to_parquet(OUTPUT_DF_PARQUET)
print(f"-> DataFrame processado salvo em '{OUTPUT_DF_PARQUET}'.")

with open(OUTPUT_MATRIX_PKL, 'wb') as f:
    pickle.dump(feature_matrix, f)
print(f"-> Matriz de features salva em '{OUTPUT_MATRIX_PKL}'.")

all_genres = sorted(df['genres'].unique().tolist())
with open(OUTPUT_GENRES_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_genres, f, ensure_ascii=False, indent=4)
print(f"-> Lista de gêneros salva em '{OUTPUT_GENRES_JSON}'.")

# Salva os encoders para consistência futura, se necessário
with open(OUTPUT_ENCODERS_PKL, 'wb') as f:
    pickle.dump({'genre_encoder': genre_encoder, 'numerical_scaler': numerical_scaler}, f)
print(f"-> Encoders salvos em '{OUTPUT_ENCODERS_PKL}'.")


print("\n--- CONSTRUÇÃO DE CACHE DE MÚSICAS CONCLUÍDA ---")
