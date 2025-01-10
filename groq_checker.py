from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def create_resume_analyzer():
    # Get API key from environment variables
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    # Initialize Groq LLM with API key
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

    # Define the expected JSON structure for evaluation results
    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "has_experience": {"type": "string"},
            "has_2_years_experience": {"type": "string"},
            "is_iitian": {"type": "string"},
            "has_deep_learning_project": {"type": "string"}
        }
    })

    # Create a prompt template with properly escaped JSON format
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the given text and evaluate the following conditions. 
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
        }}"""),
        ("user", "{input}")
    ])

    # Create the chain
    chain = prompt | llm | parser
    return chain

def analyze_resume(file_path: str) -> dict:
    try:
        # Read the content from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            context = file.read()
        
        # Create and run the analyzer
        analyzer = create_resume_analyzer()
        result = analyzer.invoke({"input": context})
        
        # Pretty print the results
        print("Analysis Results:")
        print(json.dumps(result, indent=2))
        
        return result

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except ValueError as ve:
        print(f"Configuration Error: {str(ve)}")
        return None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    analyze_resume("extracted_text.txt")