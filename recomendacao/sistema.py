import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random

# Autenticação com permissões de leitura de playlists públicas
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope="playlist-read-private playlist-read-collaborative"
))

def get_user_id():
    return sp.current_user()['id']

def get_playlists(user_id=None):
    playlists = []
    offset = 0

    if user_id is None:
        user_id = get_user_id()

    while True:
        results = sp.user_playlists(user_id, offset=offset)
        items = results['items']
        if not items:
            break
        playlists.extend(items)
        offset += len(items)
    
    return playlists

def get_tracks_from_playlists(playlists):
    track_ids = set()
    for playlist in playlists:
        playlist_id = playlist['id']
        tracks = sp.playlist_tracks(playlist_id)
        for item in tracks['items']:
            track = item['track']
            if track and track['id']:
                track_ids.add(track['id'])
    return list(track_ids)

def recomendar_musicas(track_ids, num_recomendacoes=5):
    if not track_ids:
        return []

    seed_tracks = random.sample(track_ids, min(5, len(track_ids)))
    recommendations = sp.recommendations(seed_tracks=seed_tracks, limit=num_recomendacoes)
    return recommendations['tracks']

# Altere este valor para o ID do usuário que deseja testar (ou deixe como None para usar o seu próprio perfil)
user_id_amigo = 'ID_DO_USUARIO_DO_AMIGO'  # Exemplo: 'spotify'

# Obter playlists do usuário especificado
playlists = get_playlists(user_id=user_id_amigo)

# Mostrar playlists encontradas
print("Playlists públicas do usuário:")
for playlist in playlists:
    print(f"Nome: {playlist['name']} | ID: {playlist['id']}")

# Extrair faixas e recomendar
track_ids = get_tracks_from_playlists(playlists)
recomendacoes = recomendar_musicas(track_ids)

# Mostrar recomendações
print("\nRecomendações baseadas nas playlists públicas:")
for track in recomendacoes:
    nome = track['name']
    artista = track['artists'][0]['name']
    print(f"Nome: {nome} | Artista: {artista}")
