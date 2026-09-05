# tools/spotify.py
"""
Spotify Controller - Lazy Loading (only connects when needed)
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
import time
import re
from config import settings

class SpotifyController:
    def __init__(self):
        """Initialize - NO AUTO-CONNECTION"""
        self.sp = None
        self.device_id = None
        self._connected = False
        print("🎵 Spotify module loaded (will connect when needed)")
    
    def _connect(self):
        """Lazy connect to Spotify API - only when needed"""
        if self._connected and self.sp:
            return True
        
        try:
            print("🔐 Connecting to Spotify...")
            scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing app-remote-control streaming playlist-read-private playlist-read-collaborative user-library-read user-read-playback-position"
            
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=settings.SPOTIFY_CLIENT_ID,
                client_secret=settings.SPOTIFY_CLIENT_SECRET,
                redirect_uri=settings.SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=".spotify_cache"
            ))
            
            user = self.sp.current_user()
            print(f"✅ Spotify connected as: {user['display_name']}")
            self._connected = True
            self._get_device()
            return True
            
        except Exception as e:
            print(f"❌ Spotify connection failed: {e}")
            print("💡 Make sure:")
            print("   1. You have internet connection")
            print("   2. Spotify is open in browser or desktop app")
            print("   3. You have valid credentials in config/settings.py")
            self._connected = False
            return False
    
    def _get_device(self):
        """Get the active Spotify device"""
        if not self._connected:
            return None
            
        try:
            devices = self.sp.devices()
            if devices['devices']:
                for device in devices['devices']:
                    if device['is_active']:
                        self.device_id = device['id']
                        print(f"🎵 Active device: {device['name']}")
                        return device['id']
                
                self.device_id = devices['devices'][0]['id']
                print(f"🎵 Using device: {devices['devices'][0]['name']}")
                return self.device_id
            else:
                print("⚠️ No Spotify devices found. Open Spotify in your browser first.")
                return None
        except Exception as e:
            print(f"❌ Error getting devices: {e}")
            return None
    
    def _ensure_connected(self):
        """Ensure we're connected to Spotify"""
        if not self._connected:
            return self._connect()
        return True
    
    def _ensure_device(self):
        """Ensure we have a valid device"""
        if not self._ensure_connected():
            return False
            
        if not self.device_id:
            self._get_device()
        if not self.device_id:
            print("⚠️ Please open Spotify in your browser first, then try again.")
            print("💡 Say 'open spotify' to open it.")
            return False
        return True
    
    def search_and_play_track(self, track_name, artist_name=None):
        """Search for a track and play it"""
        if not self._ensure_device():
            return False
            
        query = track_name
        if artist_name:
            query = f"{track_name} {artist_name}"
        
        try:
            results = self.sp.search(q=query, type="track", limit=5)
            
            if results['tracks']['items']:
                # Try to find exact match first
                for track in results['tracks']['items']:
                    if track_name.lower() in track['name'].lower():
                        track_uri = track['uri']
                        track_name_found = track['name']
                        artist_name_found = track['artists'][0]['name']
                        
                        print(f"▶️ Playing: {track_name_found} by {artist_name_found}")
                        self.sp.start_playback(device_id=self.device_id, uris=[track_uri])
                        return True
                
                # If no exact match, play first result
                track = results['tracks']['items'][0]
                track_uri = track['uri']
                track_name_found = track['name']
                artist_name_found = track['artists'][0]['name']
                
                print(f"▶️ Playing: {track_name_found} by {artist_name_found}")
                self.sp.start_playback(device_id=self.device_id, uris=[track_uri])
                return True
            else:
                print(f"❌ No tracks found for: {query}")
                return False
                
        except Exception as e:
            print(f"❌ Error playing: {e}")
            return False
    
    def search_and_queue_track(self, track_name, artist_name=None):
        """Search for a track and add to queue"""
        if not self._ensure_device():
            return False
            
        query = track_name
        if artist_name:
            query = f"{track_name} {artist_name}"
        
        try:
            results = self.sp.search(q=query, type="track", limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_uri = track['uri']
                track_name_found = track['name']
                artist_name_found = track['artists'][0]['name']
                
                self.sp.add_to_queue(track_uri, device_id=self.device_id)
                print(f"🎵 Queued: {track_name_found} by {artist_name_found}")
                return True
            else:
                print(f"❌ No tracks found for: {query}")
                return False
                
        except Exception as e:
            print(f"❌ Error queuing: {e}")
            return False
    
    def get_user_playlists(self):
        """Get ALL user's playlists"""
        if not self._ensure_connected():
            return []
            
        try:
            playlists = []
            results = self.sp.current_user_playlists(limit=50)
            playlists.extend(results['items'])
            
            while results['next']:
                results = self.sp.next(results)
                playlists.extend(results['items'])
            
            return playlists
        except Exception as e:
            print(f"❌ Error getting playlists: {e}")
            return []
    
    def play_playlist_by_name(self, playlist_name, only_my_playlists=False):
        """Play a playlist by name
        
        Args:
            playlist_name: Name of playlist to search for
            only_my_playlists: If True, only search user's playlists (not Spotify's entire catalog)
        """
        if not self._ensure_device():
            return False
        
        # Clean up the playlist name - REMOVE "list" from the search
        clean_name = playlist_name
        clean_name = re.sub(r'["\']', '', clean_name)  # Remove quotes
        clean_name = re.sub(r'^(my|the)\s+', '', clean_name, flags=re.IGNORECASE)  # Remove "my" or "the"
        clean_name = re.sub(r'\s+playlist$', '', clean_name, flags=re.IGNORECASE)  # Remove trailing "playlist"
        clean_name = re.sub(r'\s+list$', '', clean_name, flags=re.IGNORECASE)  # Remove trailing "list"
        clean_name = clean_name.strip()
        
        print(f"🔍 Searching for playlist: {clean_name}")
        
        # Get user's playlists
        user_playlists = self.get_user_playlists()
        
        if not user_playlists:
            print("❌ Could not fetch your playlists")
            print("💡 Make sure Spotify is connected and you have playlists")
            return False
        
        # Look for exact match in user's playlists
        for playlist in user_playlists:
            if clean_name.lower() == playlist['name'].lower():
                print(f"▶️ Playing your playlist: {playlist['name']}")
                self.sp.start_playback(device_id=self.device_id, context_uri=playlist['uri'])
                return True
        
        # Look for partial match in user's playlists
        for playlist in user_playlists:
            if clean_name.lower() in playlist['name'].lower() or playlist['name'].lower() in clean_name.lower():
                print(f"▶️ Playing your playlist: {playlist['name']}")
                self.sp.start_playback(device_id=self.device_id, context_uri=playlist['uri'])
                return True
        
        # If ONLY_MY_PLAYLISTS is True, don't search Spotify's catalog
        if only_my_playlists:
            print(f"❌ No playlist found in your library: {clean_name}")
            print("💡 Try 'list playlists' to see all your playlists")
            return False
        
        # If not found in user's playlists, search Spotify's catalog
        print("🔍 Not found in your playlists. Searching Spotify...")
        results = self.sp.search(q=clean_name, type="playlist", limit=1)
        if results['playlists']['items']:
            playlist = results['playlists']['items'][0]
            print(f"▶️ Playing playlist (public): {playlist['name']}")
            self.sp.start_playback(device_id=self.device_id, context_uri=playlist['uri'])
            return True
        else:
            print(f"❌ No playlist found for: {clean_name}")
            return False
            
    def set_volume(self, volume):
        """Set volume (0-100)"""
        if not self._ensure_device():
            return False
            
        try:
            volume = max(0, min(100, int(volume)))
            self.sp.volume(volume, device_id=self.device_id)
            print(f"🔊 Volume set to {volume}%")
            return True
        except Exception as e:
            print(f"❌ Error setting volume: {e}")
            return False
    
    def change_volume(self, amount):
        """Change volume by amount (positive or negative)"""
        if not self._ensure_device():
            return False
            
        try:
            amount = int(amount)
            
            current = self.sp.current_playback()
            if current and current['device']:
                current_volume = current['device']['volume_percent']
                new_volume = max(0, min(100, current_volume + amount))
                return self.set_volume(new_volume)
            else:
                print("⚠️ No active playback. Using default volume.")
                default_volume = 50 + amount if amount > 0 else 50
                return self.set_volume(max(0, min(100, default_volume)))
                
        except Exception as e:
            print(f"❌ Error changing volume: {e}")
            return False
    
    def pause(self):
        """Pause playback"""
        if not self._ensure_device():
            return False
            
        try:
            self.sp.pause_playback(device_id=self.device_id)
            print("⏸️ Paused")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def resume(self):
        """Resume playback"""
        if not self._ensure_device():
            return False
            
        try:
            self.sp.start_playback(device_id=self.device_id)
            print("▶️ Resumed")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def next(self):
        """Skip to next track"""
        if not self._ensure_device():
            return False
            
        try:
            self.sp.next_track(device_id=self.device_id)
            print("⏭️ Next track")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def previous(self):
        """Go to previous track"""
        if not self._ensure_device():
            return False
            
        try:
            self.sp.previous_track(device_id=self.device_id)
            print("⏮️ Previous track")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def clear_queue(self):
        """Clear the Spotify queue"""
        try:
            print("🎵 Clearing queue...")
            
            if not self._ensure_connected():
                print("ℹ️ Cannot clear queue: Spotify not connected")
                return False
            
            # Try to use the Web API to clear (by playing current track again)
            try:
                current = self.sp.current_playback()
                if current and current['item']:
                    # Play the current track again (this replaces the queue)
                    self.sp.start_playback(
                        device_id=self.device_id, 
                        uris=[current['item']['uri']]
                    )
                    print("✅ Queue cleared!")
                    return True
            except:
                pass
            
            print("✅ Queue cleared! (You may need to clear it manually in the Spotify app)")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing queue: {e}")
            return False

# Create global instance (lazy - won't connect until needed)
spotify = SpotifyController()

# ===== WRAPPER FUNCTIONS =====

def open_spotify():
    """Open Spotify in browser"""
    print("🌐 Opening Spotify in browser...")
    webbrowser.open("https://open.spotify.com")
    time.sleep(2)
    spotify._get_device()
    return True

def play_spotify_song(song_name, artist_name=None):
    """Play a song by name"""
    return spotify.search_and_play_track(song_name, artist_name)

def queue_spotify_song(song_name, artist_name=None):
    """Queue a song by name"""
    return spotify.search_and_queue_track(song_name, artist_name)

def play_spotify_playlist(playlist_name):
    """Play a playlist by name (searches your playlists + Spotify catalog)"""
    return spotify.play_playlist_by_name(playlist_name, only_my_playlists=False)

def play_my_playlist(playlist_name):
    """Play a playlist from YOUR library only"""
    return spotify.play_playlist_by_name(playlist_name, only_my_playlists=True)

def list_playlists():
    """List user's playlists"""
    playlists = spotify.get_user_playlists()
    if playlists:
        print("\n📋 Your Playlists:")
        for i, playlist in enumerate(playlists, 1):
            print(f"  {i}. {playlist['name']} ({playlist['tracks']['total']} tracks)")
        return playlists
    return []

def raise_volume(amount=10):
    """Increase volume"""
    return spotify.change_volume(amount)

def lower_volume(amount=10):
    """Decrease volume"""
    return spotify.change_volume(-amount)

def set_volume(volume):
    """Set volume to exact percentage"""
    return spotify.set_volume(volume)

def mute_volume():
    """Mute"""
    return spotify.set_volume(0)

def unmute_volume():
    """Unmute"""
    return spotify.set_volume(50)

def pause_spotify():
    """Pause"""
    return spotify.pause()

def resume_spotify():
    """Resume"""
    return spotify.resume()

def play_spotify():
    """Alias for resume"""
    return spotify.resume()

def next_track():
    """Next track"""
    return spotify.next()

def previous_track():
    """Previous track"""
    return spotify.previous()

def get_current_track():
    """Get current track"""
    return spotify.sp.current_user_playing_track() if spotify._connected else None

def clear_queue():
    """Clear queue"""
    return spotify.clear_queue()

__all__ = [
    'open_spotify',
    'play_spotify_song',
    'queue_spotify_song',
    'play_spotify_playlist',
    'play_my_playlist',
    'play_spotify',
    'list_playlists',
    'raise_volume',
    'lower_volume',
    'set_volume',
    'mute_volume',
    'unmute_volume',
    'pause_spotify',
    'resume_spotify',
    'next_track',
    'previous_track',
    'get_current_track',
    'clear_queue'
]