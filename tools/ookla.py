# tools/ookla.py
"""
Ookla Speed Test Module
Can run in background or browser with result extraction
"""

import subprocess
import re
import json
import time
import webbrowser
import threading
import speedtest
from datetime import datetime

class OoklaSpeedTest:
    def __init__(self):
        self.results = None
        self.is_running = False
        self.use_background = True  # Default to background mode
    
    def run_background_test(self):
        """
        Run speed test in background using speedtest-cli
        Returns dict with results
        """
        try:
            print("🌐 Starting speed test... (this may take 30-60 seconds)")
            self.is_running = True
            
            # Use speedtest-cli to get results
            st = speedtest.Speedtest()
            
            # Get best server
            print("  📍 Finding best server...")
            st.get_best_server()
            
            # Test download speed
            print("  📥 Testing download speed...")
            download_speed = st.download() / 1_000_000  # Convert to Mbps
            
            # Test upload speed
            print("  📤 Testing upload speed...")
            upload_speed = st.upload() / 1_000_000  # Convert to Mbps
            
            # Get ping
            ping = st.results.ping
            
            # Get server info
            server = st.results.server
            
            self.results = {
                'download': round(download_speed, 2),
                'upload': round(upload_speed, 2),
                'ping': round(ping, 2),
                'server': server['name'],
                'server_location': f"{server['country']}, {server['name']}",
                'isp': st.results.client.get('isp', 'Unknown'),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'success': True
            }
            
            self.is_running = False
            return self.results
            
        except Exception as e:
            self.is_running = False
            return {
                'success': False,
                'error': str(e),
                'message': "Speed test failed. Make sure you have internet connection."
            }
    
    def run_browser_test(self):
        """
        Run speed test in browser and try to extract results
        Falls back to background if browser method fails
        """
        try:
            print("🌐 Opening Ookla Speed Test in browser...")
            print("📊 Please wait for the test to complete...")
            
            # Open Ookla speed test
            webbrowser.open("https://www.speedtest.net")
            
            # Try to use speedtest-cli in parallel for results
            print("🔄 Running background test for results...")
            return self.run_background_test()
            
        except Exception as e:
            print(f"❌ Browser test failed: {e}")
            return self.run_background_test()
    
    def get_results(self):
        """Get formatted results for display"""
        if not self.results:
            return "No speed test results available. Run a test first."
        
        if not self.results.get('success', False):
            return f"❌ Speed test failed: {self.results.get('message', 'Unknown error')}"
        
        result_str = f"""
📊 SPEED TEST RESULTS
{'=' * 40}
📥 Download: {self.results['download']} Mbps
📤 Upload:   {self.results['upload']} Mbps
📡 Ping:     {self.results['ping']} ms
📍 Server:   {self.results['server_location']}
🏢 ISP:      {self.results['isp']}
🕐 Time:     {self.results['timestamp']}
{'=' * 40}

💡 Internet Speed Rating:
{'  🚀 Excellent' if self.results['download'] > 100 else ''}
{'  ✅ Good' if 25 <= self.results['download'] <= 100 else ''}
{'  ⚠️ Average' if 10 <= self.results['download'] < 25 else ''}
{'  ❌ Slow' if self.results['download'] < 10 else ''}
"""
        return result_str
    
    def get_speed_rating(self):
        """Get a rating for the internet speed"""
        if not self.results:
            return "Unknown"
        
        download = self.results.get('download', 0)
        
        if download > 100:
            return "🚀 Excellent (Great for 4K streaming, gaming)"
        elif download > 50:
            return "✅ Very Good (Great for HD streaming, gaming)"
        elif download > 25:
            return "✅ Good (Good for streaming, browsing)"
        elif download > 10:
            return "⚠️ Average (Basic streaming, browsing)"
        else:
            return "❌ Slow (May struggle with streaming)"

def run_speed_test(background=True):
    """
    Main function to run speed test
    
    Args:
        background: True = run in background, False = open browser
    
    Returns:
        dict with results
    """
    tester = OoklaSpeedTest()
    
    if background:
        return tester.run_background_test()
    else:
        return tester.run_browser_test()

def get_formatted_results(results):
    """Get formatted results from a results dict"""
    if not results:
        return "❌ No results available"
    
    if not results.get('success', False):
        return f"❌ Speed test failed: {results.get('message', 'Unknown error')}"
    
    result_str = f"""
📊 SPEED TEST RESULTS
{'=' * 40}
📥 Download: {results['download']} Mbps
📤 Upload:   {results['upload']} Mbps
📡 Ping:     {results['ping']} ms
📍 Server:   {results['server_location']}
🏢 ISP:      {results['isp']}
🕐 Time:     {results['timestamp']}
{'=' * 40}

💡 Internet Speed Rating:
  {'🚀 Excellent (Great for 4K streaming, gaming)' if results['download'] > 100 else ''}
  {'✅ Very Good (Great for HD streaming, gaming)' if 50 < results['download'] <= 100 else ''}
  {'✅ Good (Good for streaming, browsing)' if 25 < results['download'] <= 50 else ''}
  {'⚠️ Average (Basic streaming, browsing)' if 10 < results['download'] <= 25 else ''}
  {'❌ Slow (May struggle with streaming)' if results['download'] <= 10 else ''}
"""
    return result_str

# Simple function for quick testing
def quick_speed_test():
    """Quick speed test - returns simplified results"""
    results = run_speed_test(background=True)
    return get_formatted_results(results)

__all__ = [
    'run_speed_test',
    'get_formatted_results',
    'quick_speed_test',
    'OoklaSpeedTest'
]