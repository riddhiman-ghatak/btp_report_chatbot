import streamlit as st
import fitz
import os
import google.generativeai as genai
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if 'groq_client' not in st.session_state:
    st.session_state.groq_client = Groq()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def extract_pages_as_png(pdf_file, output_folder):
    """Extract pages from PDF as PNG images"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())
    
    pdf_document = fitz.open("temp.pdf")
    image_paths = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(output_path)
        image_paths.append(output_path)
    
    pdf_document.close()
    os.remove("temp.pdf")
    return image_paths

def prep_image(image_path):
    """Prepare image for Gemini"""
    sample_file = genai.upload_file(path=image_path)
    return sample_file

def extract_text_from_image(image_path, prompt):
    """Extract text from image using Gemini"""
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    sample_file = prep_image(image_path)
    response = model.generate_content([sample_file, prompt])
    return response.text

def create_resume_analyzer():
    """Create resume analyzer using Groq"""
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

    parser = JsonOutputParser()
    
    # Fixed prompt template with proper escaping
    prompt = ChatPromptTemplate.from_template("""
    Analyze the given text and evaluate the following conditions. 
    Return the results in JSON format with "yes" or "no" for each condition:
    
    1. Does the text mention any work experience?
    2. Does the person have at least 2 years of experience?
    3. Is the person from an IIT (Indian Institute of Technology)?
    4. Does the person have any projects in deep learning?
    
    Return exactly in this format:
    {{
        "has_experience": "yes/no",
        "has_2_years_experience": "yes/no",
        "is_iitian": "yes/no",
        "has_deep_learning_project": "yes/no"
    }}

    Text to analyze: {input}
    """)

    return prompt | llm | parser

# def get_chatbot_response(context, user_query):
#     """Get response from chatbot"""
#     client = Groq()
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that answers questions based on the provided context. Only answer questions using information from the context. If the information is not in the context, say so."
#         },
#         {
#             "role": "user",
#             "content": f"Context: {context}\n\nQuestion: {user_query}"
#         }
#     ]

#     response = client.chat.completions.create(
#         #api_key=os.getenv('GROQ_API_KEY'),
#         model="llama-3.3-70b-versatile",
#         messages=messages,
#         temperature=0.5,
#         max_tokens=1024,
#         top_p=1,
#         stream=False
#     )
#     return response.choices[0].message.content

def get_chatbot_response(context, user_query):
    """Get response from chatbot using the persistent client"""
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

    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit UI
st.set_page_config(page_title="Resume Analyzer & Chatbot", layout="wide")
st.title("Resume Analyzer & Chatbot")

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Upload PDF Resume", type=['pdf'])

if uploaded_file is not None:
    # Create containers for different processing stages
    extraction_container = st.container()
    ocr_container = st.container()
    analysis_container = st.container()

    with extraction_container:
        st.subheader("Step 1: PDF Processing")
        with st.spinner("Extracting pages from PDF..."):
            image_paths = extract_pages_as_png(uploaded_file, "output_images")
            st.success(f"✅ Successfully extracted {len(image_paths)} pages from PDF")

    with ocr_container:
        st.subheader("Step 2: OCR Processing")
        st.write("Performing OCR on extracted pages...")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract text from images
        all_extracted_text = []
        for idx, image_path in enumerate(image_paths):
            status_text.text(f"Processing page {idx + 1} of {len(image_paths)}...")
            text = extract_text_from_image(image_path, "Extract the text in the image verbatim")
            all_extracted_text.append(f"=== Page {idx + 1} ===\n{text}\n")
            progress_bar.progress((idx + 1) / len(image_paths))

        st.session_state.extracted_text = "\n".join(all_extracted_text)
        
        # Save extracted text
        with open('extracted_text.txt', 'w', encoding='utf-8') as f:
            f.write(st.session_state.extracted_text)
        
        status_text.text("✅ OCR processing complete!")
        st.success("Text extraction completed successfully!")
        
        # Show extracted text in an expander
        with st.expander("View Extracted Text"):
            st.text_area("Extracted Text", st.session_state.extracted_text, height=300)

    with analysis_container:
        st.subheader("Step 3: Resume Analysis")
        with st.spinner("Analyzing resume content..."):
            analyzer = create_resume_analyzer()
            st.session_state.analysis_results = analyzer.invoke({"input": st.session_state.extracted_text})
            st.success("✅ Analysis complete!")

# Display analysis results if available
if st.session_state.analysis_results:
    st.header("Resume Analysis Results")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Work Experience", "✅" if st.session_state.analysis_results["has_experience"] == "yes" else "❌")
    with col2:
        st.metric("2+ Years Experience", "✅" if st.session_state.analysis_results["has_2_years_experience"] == "yes" else "❌")
    with col3:
        st.metric("IIT Graduate", "✅" if st.session_state.analysis_results["is_iitian"] == "yes" else "❌")
    with col4:
        st.metric("Deep Learning Projects", "✅" if st.session_state.analysis_results["has_deep_learning_project"] == "yes" else "❌")

#Chat interface
if st.session_state.extracted_text:
    st.header("Chat with Resume")
    user_question = st.text_input("Ask a question about the resume:")
    
    if st.button("Send"):
        with st.spinner("Generating response..."):
            response = get_chatbot_response(st.session_state.extracted_text, user_question)
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Bot:** {message}")

# [Previous imports and functions remain the same]

# # Enhanced Chat Interface
# if st.session_state.extracted_text:
#     st.header("Chat with Resume")
    
#     # Initialize chat history if not exists
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
    
#     # Function to clear input
#     def clear_input():
#         st.session_state['user_input'] = ''
    
#     # Chat input area at the bottom
#     with st.container():
#         # Create a container for chat history that scrolls
#         chat_container = st.container()
        
#         # Add some spacing
#         st.markdown("<br>" * 2, unsafe_allow_html=True)
        
#         # Input area with send button side by side
#         col1, col2 = st.columns([6, 1])
#         with col1:
#             user_question = st.text_input(
#                 "", 
#                 placeholder="Ask a question about the resume...", 
#                 key="user_input",
#                 on_change=clear_input if st.session_state.get('clear_input', False) else None
#             )
#         with col2:
#             send_button = st.button("Send")

#         # Handle sending message
#         if send_button and user_question:
#             # Add user message to history
#             st.session_state.chat_history.append(("user", user_question))
            
#             # Get and add bot response
#             with st.spinner("Thinking..."):
#                 response = get_chatbot_response(st.session_state.extracted_text, user_question)
#                 st.session_state.chat_history.append(("assistant", response))
            
#             # Set flag to clear input on next rerun
#             st.session_state.clear_input = True
            
#             # Rerun to update UI
#             st.experimental_rerun()
#         else:
#             # Reset clear input flag
#             st.session_state.clear_input = False

#         # Display chat history in the container
#         with chat_container:
#             for role, message in st.session_state.chat_history:
#                 if role == "user":
#                     st.markdown(
#                         f"""
#                         <div style='background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;'>
#                             <b>You:</b> {message}
#                         </div>
#                         """, 
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     st.markdown(
#                         f"""
#                         <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px 0;'>
#                             <b>Assistant:</b> {message}
#                         </div>
#                         """, 
#                         unsafe_allow_html=True
#                     )

#         # Add a clear chat button
#         if st.button("Clear Chat"):
#             st.session_state.chat_history = []
#             st.experimental_rerun()