# tools/youtube.py
"""
YouTube-related functions for the AI Assistant
"""

import yt_dlp
import webbrowser
import re
import subprocess
import os

class YouTubeSession:
    """Manages YouTube search session state"""
    def __init__(self):
        self.results = []
        self.current_search = None
        
    def clear(self):
        self.results = []
        self.current_search = None
        
    def has_results(self):
        return len(self.results) > 0
        
    def get_result(self, index):
        if 0 <= index < len(self.results):
            return self.results[index]
        return None
        
    def get_titles(self):
        return [f"{i+1}. {video['title']}" for i, video in enumerate(self.results)]

# Global session instance
youtube_session = YouTubeSession()

def open_youtube(search_query=None):
    """Opens YouTube in browser"""
    if search_query:
        url = f"https://www.youtube.com/results?search_query={search_query}"
    else:
        url = "https://www.youtube.com/"
    webbrowser.open(url)
    print("✅ Opened YouTube")
    return True

def search_youtube(search_query, max_results=5):
    """Searches YouTube and stores results"""
    print(f"🔍 Searching YouTube for: {search_query}")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = ydl.extract_info(f"ytsearch{max_results}:{search_query}", download=False)
            
            if search_results and 'entries' in search_results:
                youtube_session.results = search_results['entries']
                youtube_session.current_search = search_query
                
                print(f"✅ Found {len(youtube_session.results)} videos:")
                for i, video in enumerate(youtube_session.results, 1):
                    title = video.get('title', 'Unknown title')
                    duration = video.get('duration', 0)
                    duration_str = f"{duration//60}:{duration%60:02d}" if duration else "N/A"
                    print(f"  {i}. {title} ({duration_str})")
                
                return youtube_session.results
            else:
                print("❌ No results found")
                return []
                
    except Exception as e:
        print(f"❌ Error searching YouTube: {e}")
        return []

def play_youtube_video(index=None, autoplay=True):
    """Plays a YouTube video from current search results"""
    if not youtube_session.has_results():
        print("❌ No search results. Please search first.")
        return False
    
    if index is None:
        index = 0
    else:
        index = index - 1
    
    video = youtube_session.get_result(index)
    if video:
        video_url = f"https://www.youtube.com/watch?v={video['id']}"
        if autoplay:
            video_url += "&autoplay=1"
        title = video.get('title', 'Unknown title')
        
        print(f"▶️ Playing: {title}")
        webbrowser.open(video_url)
        return True
    else:
        print(f"❌ Invalid video number. Please choose 1-{len(youtube_session.results)}")
        return False

def show_youtube_links():
    """Shows full URLs of all videos in the current search"""
    if not youtube_session.has_results():
        print("❌ No search results. Please search first.")
        return
    
    print(f"\n🔗 Video Links for '{youtube_session.current_search}':")
    for i, video in enumerate(youtube_session.results, 1):
        video_url = f"https://www.youtube.com/watch?v={video['id']}"
        title = video.get('title', 'Unknown title')
        print(f"  {i}. {title}")
        print(f"     🔗 {video_url}")

def show_youtube_info(index):
    """Shows detailed info about a specific video"""
    if not youtube_session.has_results():
        print("❌ No search results. Please search first.")
        return
    
    video = youtube_session.get_result(index - 1)
    if video:
        print(f"\n📋 Video Info:")
        print(f"  Title: {video.get('title', 'Unknown')}")
        print(f"  URL: https://www.youtube.com/watch?v={video['id']}")
        print(f"  Duration: {video.get('duration', 'N/A')} seconds")
        print(f"  Uploader: {video.get('uploader', 'Unknown')}")
        print(f"  Views: {video.get('view_count', 'N/A')}")
    else:
        print(f"❌ Invalid video number. Please choose 1-{len(youtube_session.results)}")

def clear_youtube_search():
    """Clears current search results"""
    youtube_session.clear()
    print("🧹 Search results cleared")

__all__ = [
    'open_youtube',
    'search_youtube',
    'play_youtube_video',
    'show_youtube_links',
    'show_youtube_info',
    'clear_youtube_search',
    'youtube_session'
]