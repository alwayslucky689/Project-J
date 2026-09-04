import json
import webbrowser
import requests
import subprocess

def handle_command(user_input):
    """Process the given command/question request to determine what actions to take, and return it as a LIST of actions explicitly"""
    prompt=prompt=f"""
        You are an AI that controls the user's computer.
        You MUST respond with ONLY a JSON array of actions.
        Available tools:
        -ask_question: If the user asks a question (starts with what/why/how/when/where/who/is/are/do/does or ends with ?), use "ask_question" and answer the question naturally
        - open_discord: Opens Discord as a process on my computer.
        - open_youtube: Opens Youtube. Takes an optional "search_query" string
        - open_spotify: Opens Spotify.
        - no_tool: Use this when users request does not match any tool. Takes "message" explaining you don't understand.
        Examples:
        User: "Open YouTube and search for John Peck"
        Assistant: {{"tool": "open_youtube", "search_query": "John Peck"}}
        User: "Open YouTube"
        Assistant: {{"tool": "open_youtube"}}
        User: "Open Spotify"
        Assistant: {{"tool": "open_spotify"}}
        User: "Open Discord"
        Assistant: {{"tool": "open_discord"}}
        User: "Open YouTube and open Spotify"
        Assistant: [
        {{"tool": "open_youtube"}},
        {{"tool": "open_spotify"}}
        ]
        User: "Open YouTube and search for cats"
        Assistant: [
        {{"tool": "open_youtube", "search_query": "cats"}}
        ]  
         User: "Open YouTube and what is the weather?"
        Assistant: [
        {{"tool": "open_youtube"}},
        {{"tool": "ask_question", "question": "What is the weather today?"}}
        ]
        User: "How big is the sun?"
        Assistant: [
        {{"tool": "ask_question", "question": "How big is the sun?"}}
        ]
        User: "{user_input}"
        Assistant:
        """
    response = requests.post(
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
    #ai_output = response.json()["response"].strip()
    
    try:
        # Try to parse as list of actions
        actions = json.loads(ai_output)
        # Ensure it's a list (if AI outputs a single action, wrap it)
        if isinstance(actions, dict):
            actions = [actions]
        return actions
    except json.JSONDecodeError:
        return [{"tool": "error", "message": f"Invalid JSON: {ai_output}"}]
def execute_tool(ai_decision):
    """Runs one or more tools based on the AI's decision."""
    # If it's a list of actions (new format)
    if isinstance(ai_decision, list):
        for action in ai_decision:
            execute_single_tool(action)
    else:
        # Backwards compatibility for single actions
        execute_single_tool(ai_decision)
def execute_single_tool(action):
    """Executes a single tool action."""
    tool_name=action.get("tool")
    if tool_name=="open_youtube":
        search_query=action.get("search_query")
        open_youtube(search_query)
    elif tool_name=="open_spotify":
        open_spotify()
    elif tool_name=="open_discord":
            open_discord()
    elif tool_name=="no_tool":
        print("IDK WHAT U WANT BRUH")
    elif tool_name=="ask_question":
        question = action.get("question")
        answer = ask_question(question)
        print(f"🤖 {answer}\n")
    else:
        print("what??")
    
def ask_question(question):
    """
    Step 3: Answer a question naturally, without JSON constraints.
    """
    question_prompt = f"""
    You are a helpful AI assistant, that is also a little sarcastic and adds quippy jokes. Answer the user's question naturally.
    Be accurate. Don't mention that you're an AI.
    
    User: {question}
    Assistant:
    """
    
    response = requests.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b-instruct-q4_K_M",
            "prompt": question_prompt,
            "stream": False,
            "temperature": 0.7
        }
    )
    print("Question answered")
    
    return response.json()["response"].strip()
    
       
def open_youtube(search_query=None):
    """
    Opens YouTube in browser. If a search query is provided, 
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
def open_discord():
    subprocess.Popen(r"C:\Users\pstef\AppData\Local\Discord\Update.exe --processStart Discord.exe",shell=True)



    
while True:
    user_input=input("What do you want? > ")
    if user_input.lower() in ["quit","bye","exit","cao","kys"]:
        break
    #ai_dec=ask_ollama(user_input)
    actions=handle_command(user_input)
    execute_tool(actions)
    
print("kms ig")