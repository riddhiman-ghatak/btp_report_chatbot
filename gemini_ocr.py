"""

conda install -c conda-forge google-generativeai

"""
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import os

load_dotenv()

# Get the API key
API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)

def prep_image(image_path):
    # Upload the file and print a confirmation.
    sample_file = genai.upload_file(path=image_path,
                                display_name="Diagram")
    print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
    file = genai.get_file(name=sample_file.name)
    print(f"Retrieved file '{file.display_name}' as: {sample_file.uri}")
    return sample_file

def extract_text_from_image(image_path, prompt):
    # Choose a Gemini model.
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([image_path, prompt])
    return response.text


output_dir = 'output_images'
all_extracted_text = []

# Get all PNG files from the directory
image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(output_dir, image_file)
    print(f"\nProcessing {image_file}...")
    
    sample_file = prep_image(image_path)
    text = extract_text_from_image(sample_file, "Extract the text in the image verbatim")
    
    if text:
        all_extracted_text.append(f"=== Text from {image_file} ===\n{text}\n\n")
    else:
        print(f"Failed to extract text from {image_file}")

# Save all extracted text to a file
with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.writelines(all_extracted_text)

print("\nExtraction complete! Results saved to extracted_text.txt")