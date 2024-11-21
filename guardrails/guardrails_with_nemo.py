from typing import List
from nemoguardrails import LLMRails, RailsConfig
from langchain.chat_models import ChatOpenAI

def create_rails_config():
    """Create and return the NeMo Guardrails configuration"""
    config = RailsConfig.from_content(
        # Main configuration
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-3.5-turbo"
                }
            ],
            "rails": {
                "input": {
                    "flows": ["self check input"]
                },
                "output": {
                    "flows": ["self check output"]
                }
            }
        },
        # Prompts for the rails
        prompts=[
            {
                "task": "self_check_input",
                "content": """Your task is to check if the student question below is related to computer science.
                
                Student question: {{ user_input }}
                
                Question: Should this question be allowed (Yes/No)?
                Consider:
                - Allow only computer science related questions
                - Questions about programming, algorithms, data structures, etc. are allowed
                - General questions unrelated to CS should be blocked
                
                Answer:"""
            },
            {
                "task": "self_check_output",
                "content": """Your task is to check if the tutor's response follows Socratic teaching principles.

                Student question: {{ user_input }}
                Tutor response: {{ bot_response }}

                Check if the response:
                1. Does NOT reveal the direct answer
                2. Uses probing questions to guide thinking
                3. Helps break down the concept
                4. Encourages critical thinking

                Question: Should this response be blocked (Yes/No)?
                Answer:"""
            }
        ],
        # Colang files for dialog flows
        colang_files=["""
            define user ask cs question
                "How does binary search work?"
                "What is recursion?"
                "Explain data structures"

            define bot refuse to respond
                "I apologize, but I can only answer questions related to computer science."

            define bot express cannot help
                "I apologize, but I'm having trouble generating an appropriate response. Please try rephrasing your question."

            define flow
                user ask cs question
                bot respond socratically

            define bot respond socratically
                "Let me help guide you through this concept with some questions..."
        """]
    )
    return config

def get_socratic_response(query: str, max_attempts: int = 3) -> str:
    """Get a valid Socratic response for a query using NeMo Guardrails"""
    
    # Initialize the guardrails
    config = create_rails_config()
    rails = LLMRails(config)
    
    # Generate response with guardrails
    try:
        response = rails.generate(messages=[{
            "role": "user",
            "content": query
        }])
        return response["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response. Please try again."

def main():
    # Sample student queries
    sample_queries = [
        "How does a binary search algorithm work?",
        "What is the difference between HTTP and HTTPS?",
        "What is your favorite color?",
        "Can you explain recursion in programming?",
        "How do I implement a stack data structure?"
    ]
    
    print("Processing sample queries with NeMo Guardrails...\n")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        response = get_socratic_response(query)
        print(f"Response: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main() 