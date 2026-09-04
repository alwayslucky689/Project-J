import json
import webbrowser
import requests

def open_youtube(search_query=None):
    """
    Opens YouTube in your browser. If a search query is provided, 
    it searches for that term.
    """
    if search_query:
        url=f"https://www.youtube.com/results?search_query={search_query}"
    else:
        url="https://www.youtube.com/"
    
    webbrowser.open(url)
    print("Opened youtube")
def open_spotify():
    url2="https://open.spotify.com/"
    webbrowser.open(url2)
    print("Opened spotify")
def ask_ollama(user_input):
    """
    Sends the user's request to Ollama, and returns the AI's response as a 
    Python dictionary.
    """
    prompt=f"""
    You are an AI assistant on my computer that executes commands and answers my questions
    You MUST respond with ONLY a JSON object.
    Available tools:
    - open_youtube: Opens Youtube. Takes an optional "search_query" string
    - open_spotify: Opens Spotify.
    - no_tool: Use this when users request does not match any tool. Takes "message" explaining you don't understand.
    Examples:
    User: "Open YouTube and search for metal music"
    Assistant: {{"tool": "open_youtube", "search_query": "john peck"}}
    
    User: "Open YouTube"
    Assistant: {{"tool": "open_youtube"}}
    User: "Open Spotify"
    Assistant: {{"tool": "open_spotify"}}
    User: "{user_input}"
    Assistant:
    """
    response=requests.post(
        url="http://localhost:11434/api/generate",
        json={
                "model": "llama3.1:8b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False
        }
    )

    response_json = response.json()
    ai_output = response_json["response"]
    ai_output = ai_output.strip()
    try:
        result=json.loads(ai_output)
        return result
    except json.JSONDecodeError:
        return {"tool":"error","message":f"AI output failed: {ai_output}"}
while True:
    user_input=input("What do you want? > ")
    if user_input.lower() in ["quit","bye","exit","cao","kys"]:
        break
    ai_dec=ask_ollama(user_input)
    tool_name=ai_dec.get("tool")
    if tool_name=="open_youtube":
        search_query=ai_dec.get("search_query")
        open_youtube(search_query)
    elif tool_name=="open_spotify":
        open_spotify()
    else:
        print("IDK WHAT U WANT BRUH")
print("kms ig")