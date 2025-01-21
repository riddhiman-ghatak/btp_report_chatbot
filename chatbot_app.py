import os
import streamlit as st
from streamlit_chat import message
from groq import Groq

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def read_context_file(file_path):
    """Read the context from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        st.error(f"Error reading context file: {e}")
        return None

def create_chatbot():
    """Initialize the Groq client."""
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("Error: GROQ_API_KEY environment variable not set")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

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

# Streamlit App
def main():
    st.set_page_config(page_title="AI Chatbot", layout="wide")
    st.title("💬 AI Chatbot")
    
    # Sidebar for context
    st.sidebar.title("Context Setup")
    context_file = st.sidebar.file_uploader("Upload Context File (text format)", type=["txt"])
    temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.5)

    # Initialize chatbot
    client = create_chatbot()
    context = ""
    if context_file:
        context = read_context_file(context_file.name)
        if context:
            st.sidebar.success("Context successfully loaded!")

    if not context:
        st.sidebar.warning("Please upload a context file to start.")
        return

   
    st.write("---")
    user_input = st.text_input("Type your message here:", key="input")
    if st.button("Send", key="send"):
        if user_input.strip():
            # Add user message to session state
            st.session_state["messages"].append({"role": "user", "content": user_input, "id": len(st.session_state["messages"])})
            
            # Get chatbot response
            response = get_chatbot_response(client, context, user_input.strip(), temperature)
            st.session_state["messages"].append({"role": "bot", "content": response, "id": len(st.session_state["messages"])})
        else:
            st.warning("Please enter a valid message.")

    # Chat History
    st.write("### Chat History")
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{msg['id']}")
        else:
            message(msg["content"], is_user=False, key=f"bot_{msg['id']}")

if __name__ == "__main__":
    main()
