import pytesseract as pt

# Extract text from the image
text = pt.image_to_string('/content/output_images/page_1.png', lang='eng')

# Define the path to save the text file
output_file_path = '/content/output_text.txt'

# Write the extracted text to a .txt file
with open(output_file_path, 'w') as file:
    file.write(text)

print(f"Text has been saved to {output_file_path}")
