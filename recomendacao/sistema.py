import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Configurações do app ---
CLIENT_ID = '0db6eab6b5674742bf395225bdbede8e'
CLIENT_SECRET = 'ae4e1f84b4b9424087eb593d4130bbd9'
REDIRECT_URI = 'http://127.0.0.1:8888/callback'
SCOPE = 'playlist-read-private user-library-read user-read-private'

# --- Autenticação via OAuth2 ---
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
))

# --- Pega informações do usuário ---
user = sp.current_user()
print(f"Usuário logado: {user['display_name']} (ID: {user['id']})")

# --- Lista playlists do usuário ---
playlists = sp.current_user_playlists()
print(f"\nPlaylists encontradas: {len(playlists['items'])}")

for i, playlist in enumerate(playlists['items']):
    print(f"{i+1}. {playlist['name']} (ID: {playlist['id']})")

# --- Pega faixas da primeira playlist (se existir) ---
if playlists['items']:
    playlist_id = playlists['items'][0]['id']
    results = sp.playlist_tracks(playlist_id)

    print(f"\n🎵 Músicas da playlist '{playlists['items'][0]['name']}':")
    for i, item in enumerate(results['items']):
        track = item['track']
        print(f"{i+1}. {track['name']} - {track['artists'][0]['name']}")
