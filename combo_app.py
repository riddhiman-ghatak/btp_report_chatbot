import os
import streamlit as st
from streamlit_chat import message
import fitz
import google.generativeai as genai
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize Groq client
def create_groq_client():
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("Error: GROQ_API_KEY environment variable not set")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

# Extract pages from PDF as PNG images
def extract_pages_as_png(pdf_file):
    """Extract pages from PDF as PNG images."""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    image_paths = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        output_path = f"page_{page_num + 1}.png"
        pix.save(output_path)
        image_paths.append(output_path)
    pdf_document.close()
    return image_paths

# Prepare image for Gemini
def prep_image(image_path):
    """Prepare image for Gemini."""
    sample_file = genai.upload_file(path=image_path)
    return sample_file

# Extract text from image using Gemini
def extract_text_from_image(image_path, prompt):
    """Extract text from image using Gemini."""
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    sample_file = prep_image(image_path)
    response = model.generate_content([sample_file, prompt])
    return response.text

# Create resume analyzer
def create_resume_analyzer():
    """Create resume analyzer using Groq."""
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
    parser = JsonOutputParser()
    # prompt = ChatPromptTemplate.from_template("""
    # Analyze the given text and evaluate the following conditions.
    # Return the results in JSON format with "yes" or "no" for each condition:
    
    # 1. Does the text mention any work experience?
    # 2. Does the person have at least 2 years of experience?
    # 3. Is the person from an IIT (Indian Institute of Technology)?
    # 4. Does the person have any projects in deep learning?
    
    # Return exactly in this format:
    # {{
    #     "has_experience": "yes/no",
    #     "has_2_years_experience": "yes/no",
    #     "is_iitian": "yes/no",
    #     "has_deep_learning_project": "yes/no"
    # }}

    # Text to analyze: {input}
    # """)

    prompt = ChatPromptTemplate.from_template("""
    Analyze the given research report and evaluate the following conditions.
    Return the results in JSON format with "yes" or "no" for each condition:
    
    1. Does the report include an acknowledgement section?
    2. Does the report contain an introduction section?
    3. Is the report based on transformer models?
    4. Does the report specifically mention attention layers?
    5. Is the report published by a well-known company or institution?
    
    Return exactly in this format:
    {{
        "has_acknowledgement": "yes/no",
        "has_introduction": "yes/no",
        "is_transformer_based": "yes/no",
        "mentions_attention_layer": "yes/no",
        "published_by_famous_entity": "yes/no"
    }}

    Research report to analyze: {input}
    """)

    return prompt | llm | parser

# Get chatbot response
def get_chatbot_response(client, context, user_query, temperature=0.5):
    """Generate response from chatbot."""
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
    st.set_page_config(page_title="PDF Analysis and Chatbot", layout="wide")
    st.title("📄 PDF Analysis and AI Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "extracted_text" not in st.session_state:
        st.session_state["extracted_text"] = None
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None

    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    client = create_groq_client()

    if uploaded_file:
        # PDF Processing
        st.subheader("Step 1: PDF Processing")
        with st.spinner("Extracting pages from PDF..."):
            image_paths = extract_pages_as_png(uploaded_file)
            st.success(f"✅ Successfully extracted {len(image_paths)} pages from the PDF.")

        # OCR Processing
        st.subheader("Step 2: Text Extraction")
        with st.spinner("Extracting text from images..."):
            all_extracted_text = []
            progress_bar = st.progress(0)
            for idx, image_path in enumerate(image_paths):
                text = extract_text_from_image(image_path, "Extract the text in the image verbatim")
                all_extracted_text.append(f"=== Page {idx + 1} ===\n{text}\n")
                progress_bar.progress((idx + 1) / len(image_paths))
            st.session_state.extracted_text = "\n".join(all_extracted_text)
            st.success("✅ Text extraction completed!")
            with st.expander("View Extracted Text"):
                st.text_area("Extracted Text", st.session_state.extracted_text, height=300)

        # Resume Analysis
        st.subheader("Step 3: Resume Analysis")
        with st.spinner("Analyzing resume content..."):
            analyzer = create_resume_analyzer()
            st.session_state.analysis_results = analyzer.invoke({"input": st.session_state.extracted_text})
            st.success("✅ Analysis complete!")
            # Display analysis results
            st.header("Resume Analysis Results")
            analysis = st.session_state.analysis_results
            # for label, key in [
            #     ("Work Experience", "has_experience"),
            #     ("2+ Years Experience", "has_2_years_experience"),
            #     ("IIT Graduate", "is_iitian"),
            #     ("Deep Learning Projects", "has_deep_learning_project")
            # ]:
            for label, key in [
                ("Acknowledgement Section", "has_acknowledgement"),
                ("Introduction Section", "has_introduction"),
                ("Based on Transformer Models", "is_transformer_based"),
                ("Mentions Attention Layer", "mentions_attention_layer"),
                ("Published by Famous Entity", "published_by_famous_entity")
            ]:

                st.write(f"{label}: {'✅' if analysis[key] == 'yes' else '❌'}")

    # Chatbot Section
    if st.session_state.extracted_text and client:
        st.subheader("Step 4: Chat with Extracted Content")
        temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.5)
        user_input = st.text_input("Type your message here:", key="input")
        if st.button("Send", key="send"):
            if user_input.strip():
                st.session_state["messages"].append({"role": "user", "content": user_input, "id": len(st.session_state["messages"])})
                response = get_chatbot_response(client, st.session_state.extracted_text, user_input.strip(), temperature)
                st.session_state["messages"].append({"role": "bot", "content": response, "id": len(st.session_state["messages"])})
            else:
                st.warning("Please enter a valid message.")

        # Display chat history
        st.write("### Chat History")
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{msg['id']}")
            else:
                message(msg["content"], is_user=False, key=f"bot_{msg['id']}")

if __name__ == "__main__":
    main()
