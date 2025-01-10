from PyPDF2 import PdfReader


reader = PdfReader('test1.pdf')


text = ""
for page in reader.pages:
    text += page.extract_text()


with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)