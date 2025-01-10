import pytesseract as pt


text = pt.image_to_string('/content/output_images/page_1.png', lang='eng')


output_file_path = '/content/output_text.txt'


with open(output_file_path, 'w') as file:
    file.write(text)

print(f"Text has been saved to {output_file_path}")
