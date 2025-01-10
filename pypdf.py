from PyPDF2 import PdfReader

# Create PDF reader object
reader = PdfReader('test1.pdf')

# Extract text from all pages
text = ""
for page in reader.pages:
    text += page.extract_text()

# Save to file
with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)