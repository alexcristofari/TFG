import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random

# Definindo variáveis de autenticação
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="ID",
                                               client_secret="SECRET",
                                               redirect_uri="URI",
                                               scope="user-library-read playlist-read-private"))

def obter_playlists_do_usuario():
    playlists = sp.current_user_playlists()
    minhas_playlists = []

    for playlist in playlists['items']:
        if playlist['owner']['id'] == sp.current_user()['id']:
            minhas_playlists.append(playlist)

    return minhas_playlists

def obter_musicas_da_playlist(playlist_id):
    faixas = []
    resultados = sp.playlist_tracks(playlist_id)

    for item in resultados['items']:
        musica = item['track']
        faixas.append({
            'nome': musica['name'],
            'artista': musica['artists'][0]['name'],
            'id': musica['id']
        })

    return faixas

def recomendar_musicas(playlists):
    todas_as_musicas = []

    # Coletar todas as músicas de minhas playlists
    for playlist in playlists:
        faixas = obter_musicas_da_playlist(playlist['id'])
        todas_as_musicas.extend(faixas)

    # Aqui, você pode usar qualquer algoritmo para recomendar as faixas
    # Como exemplo, vamos apenas recomendar aleatoriamente algumas faixas
    faixas_recomendadas = random.sample(todas_as_musicas, 5)  # Recomendando 5 faixas

    return faixas_recomendadas

# Fluxo principal
minhas_playlists = obter_playlists_do_usuario()
faixas_recomendadas = recomendar_musicas(minhas_playlists)

# Exibindo as recomendações
print("Recomendações baseadas nas suas playlists:")
for faixa in faixas_recomendadas:
    print(f"Nome: {faixa['nome']} | Artista: {faixa['artista']}")
