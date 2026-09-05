import json
import requests
from config.config_loader import load_config
from config.personality_manager import PersonalityManager
from tools import web_tools, app_tools, memory_tools, personality_tools
from memory.fact_memory import get_facts_context

# Initialize components
config = load_config()
personality_mgr = PersonalityManager()
personality_tools.init_personality_tools(personality_mgr)

# ... (Keep your existing tool function definitions) ...

def build_system_prompt():
    """Builds the system prompt with current personality and memory."""
    system_prompt = personality_mgr.get_current_prompt("system")
    facts = get_facts_context()
    if facts:
        system_prompt += f"\n\n{facts}"
    return system_prompt

def handle_command(user_input):
    """Uses the current personality's command prompt."""
    command_prompt = personality_mgr.get_current_prompt("command")
    
    # Inject the user input into the prompt
    full_prompt = command_prompt.replace("{user_input}", user_input)
    
    response = requests.post(
        url=config["ollama"]["url"],
        json={
            "model": config["ollama"]["model"],
            "prompt": full_prompt,
            "stream": False
        }
    )
    # ... (rest of your existing JSON parsing logic) ...
    return json.loads(response.json()["response"])

def handle_question(user_input):
    """Uses the current personality's question prompt."""
    question_prompt = personality_mgr.get_current_prompt("question")
    full_prompt = question_prompt.replace("{user_input}", user_input)
    
    response = requests.post(
        url=config["ollama"]["url"],
        json={
            "model": config["ollama"]["model"],
            "prompt": full_prompt,
            "stream": False
        }
    )
    return response.json()["response"].strip()

def main():
    print("🤖 AI Assistant Ready. Say a wake word to change personality.")
    print(f"Current personality: {personality_mgr.get_current_name()}")
    print("-" * 40)
    
    while True:
        user_input = input("\n🎤 You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break
        
        # 1. Check for personality switch (wake word)
        detected = personality_mgr.detect_personality(user_input)
        if detected and personality_mgr.switch_to(detected):
            print(f"🧠 Switched personality to: {detected}")
            # Remove the wake word from the input before processing
            for word in personality_mgr.personalities[detected]["wake_words"]:
                user_input = user_input.replace(word, "").strip()
            if not user_input:  # If they only said the wake word, continue loop
                print(f"💬 {detected} personality active. How can I help?")
                continue
        
        # 2. Process the command/question using the current personality
        # (Your existing logic: classify, then handle_command or handle_question)
        # ... (Insert your existing classification and execution logic here) ...

if __name__ == "__main__":
    main()