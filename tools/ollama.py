# tools/ollama.py
"""
Ollama/AI-related functions for the AI Assistant
"""
from config import settings
import requests
import json
import re

def ask_question(question):
    """Sends a question to Ollama and returns a natural language answer"""
    print(f"🤔 Answering: {question}")
    
    question_prompt = f"""
    You are a helpful AI assistant. Answer the user's question naturally and accurately.
    Be conversational. Don't mention that you're an AI.
    
    User: {question}
    Assistant:"""
    
    try:
        response = requests.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": settings.REASONING_MODEL,  # Use reasoning model
            "prompt": question_prompt,
            "stream": False,
            "temperature": 0.7
        },
        timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            
            if not answer:
                return "I couldn't generate a response. Please try again."
            
            return answer
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Ollama is taking too long. Try again."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running."
    except Exception as e:
        return f"Error: {str(e)}"

def is_question(text):
    """Check if text is a question"""
    text = text.strip()
    if text.endswith("?"):
        return True
    
    question_words = ["what", "why", "how", "when", "where", "who", "which", 
                      "does", "do", "is", "are", "did", "could", "would", 
                      "should", "will", "can", "tell me", "explain", 
                      "describe", "define", "meaning of", "definition of"]
    
    lower_text = text.lower()
    for word in question_words:
        if lower_text.startswith(word) or f" {word} " in lower_text:
            return True
    
    return False

__all__ = [
    'ask_question',
    'is_question'
]
