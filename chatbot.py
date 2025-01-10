import os
from groq import Groq
import sys

def read_context_file(file_path):
    """Read the context from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def create_chatbot():
    """Initialize the Groq client."""
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("Error: GROQ_API_KEY environment variable not set")
            sys.exit(1)
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        sys.exit(1)

def get_chatbot_response(client, context, user_query, temperature=0.5):
    """Get response from the chatbot based on context and query."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the "
                    "provided context. Only answer questions using information from "
                    "the context. If the information is not in the context, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {user_query}"
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e}"

def main():
    # Initialize the chatbot
    client = create_chatbot()
    
    # Read the context file
    context = read_context_file('extracted_text.txt')
    
    print("Chatbot initialized! Type 'quit' to exit.")
    print("-" * 50)

    # Main chat loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            print("Please enter a question.")
            continue
        
        # Get and print the response
        response = get_chatbot_response(client, context, user_input)
        print("\nChatbot:", response)

if __name__ == "__main__":
    main()