# assistant.py

import json
import requests
import os
import sys

# Import your settings and modules
from config import settings
from config.personality_manager import PersonalityManager
from tools import youtube,spotify,discord,ollama,ookla, personality_tools
from memory.fact_memory import get_facts_context

# 1. Initialize Personality Manager
personality_mgr = PersonalityManager()
personality_tools.init_personality_tools(personality_mgr)

# 2. Import your existing tool functions
# (Assuming you have functions like open_website, open_application, remember_fact etc.)
# If they are in separate files, ensure they are imported.
# Example: from tools.web_tools import open_website, search_web
# from tools.app_tools import open_application
# from tools.memory_tools import remember_fact

# 3. Helper: Build the full system prompt with memory
def build_system_prompt():
    """Combines the personality system prompt with stored facts."""
    base_system = personality_mgr.get_prompt("system")
    facts = get_facts_context()
    if facts:
        # Inject facts into the prompt
        return base_system.replace("{user_facts}", facts)
    return base_system.replace("{user_facts}", "")

# 4. Core AI Interaction Functions
def ask_ollama(prompt_text, is_json=False):
    """
    Sends a prompt to Ollama and returns the response.
    If is_json is True, it attempts to parse the response as JSON.
    """
    full_prompt = prompt_text
    if is_json:
        # Ensure the model knows to output JSON
        full_prompt = f"{full_prompt}\n\nYou MUST respond with ONLY valid JSON. Do not include any other text."

    response = requests.post(
        url=settings.OLLAMA_URL,  # From your settings.py
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    
    if response.status_code != 200:
        return {"error": f"Ollama error: {response.status_code}"}
    
    ai_output = response.json().get("response", "").strip()
    
    if is_json:
        try:
            return json.loads(ai_output)
        except json.JSONDecodeError:
            return {"tool": "error", "message": f"Invalid JSON: {ai_output}"}
    
    return ai_output

# 5. Command Router
def route_request(user_input):
    """
    Handles the user input: checks for wake word, then routes to command or question.
    Returns a string (for questions) or executes the tool (for commands).
    """
    # Step A: Check for personality switch (wake word)
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        # Remove the wake word from the input
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
        print(f"🧠 Switched personality to: {detected}")
        # If only the wake word was said, return.
        if not user_input:
            return f"💬 {detected.capitalize()} personality active. How can I help?"

    # Step B: Determine if it's a command or question
    # We can use a simple heuristic (presence of action words) or a cheap classifier.
    # For simplicity, let's check if it looks like an action:
    action_keywords = ["open", "play", "search", "start", "run", "remember", "switch", "change"]
    is_command = any(word in user_input.lower() for word in action_keywords)
    
    if is_command:
        # Get the command prompt for the current personality
        command_template = personality_mgr.get_prompt("command")
        # The command prompt should have a {user_input} placeholder
        prompt = command_template.replace("{user_input}", user_input)
        
        # Add the system prompt context to guide the AI
        system_context = build_system_prompt()
        full_prompt = f"{system_context}\n\n{prompt}"
        
        decision = ask_ollama(full_prompt, is_json=True)
        
        # Execute the tool(s)
        return execute_tool(decision)
    else:
        # It's a question
        question_template = personality_mgr.get_prompt("question")
        prompt = question_template.replace("{user_input}", user_input)
        system_context = build_system_prompt()
        full_prompt = f"{system_context}\n\n{prompt}"
        
        return ask_ollama(full_prompt, is_json=False)

# 6. Tool Executor (Refactored to handle your existing tools)
def execute_tool(decision):
    """
    Executes the tool specified in the AI's JSON decision.
    Supports both single actions and lists of actions.
    """
    # If it's a list, execute each one sequentially
    if isinstance(decision, list):
        results = []
        for action in decision:
            results.append(execute_single_action(action))
        return "\n".join(results)
    else:
        return execute_single_action(decision)

def execute_single_action(action):
    tool_name = action.get("tool")
    
    # Security check: Is this tool allowed?
    if tool_name not in settings.ALLOWED_TOOLS:
        return f"⚠️ Security Error: Tool '{tool_name}' is not in the allowed list."
    
    # Route to the appropriate function
    # Map tool names to actual functions (you'll need to import these)
    # Example mapping:
    if tool_name == "open_website":
        return web_tools.open_website(action.get("url"))
    elif tool_name == "search_web":
        return web_tools.search_web(action.get("query"), action.get("engine", "google"))
    elif tool_name == "open_application":
        return app_tools.open_application(action.get("app_name"))
    elif tool_name == "remember_fact":
        return memory_tools.remember_fact(action.get("fact"))
    elif tool_name == "change_personality":
        return personality_tools.change_personality(action.get("name"))
    elif tool_name == "error":
        return f"⚠️ AI Error: {action.get('message')}"
    else:
        return f"⚠️ Unknown tool: {tool_name}. AI said: {action}"

# 7. The Main Loop
def main():
    print("🤖 Project-J AI Assistant Ready!")
    print(f"Current personality: {personality_mgr.get_current_name()}")
    print(f"Wake words: {personality_mgr.get_wake_words()}")
    print("Type 'quit' to exit.")
    print("-" * 40)
    
    while True:
        user_input = input("\n🎤 You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break
        
        result = route_request(user_input)
        print(f"🤖 {result}")

if __name__ == "__main__":
    main()