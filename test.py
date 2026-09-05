import requests
import json

# CORRECT - using HTTP, not HTTPS
url = "http://localhost:11434/api/generate"
# OR
url = "http://127.0.0.1:11434/api/generate"

payload = {
    "model": "llama3.2",  # Make sure you have this model
    "prompt": "Hello, world!",
    "stream": False
}

response = requests.post(
    url,
    json=payload,
    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
print(response.json())