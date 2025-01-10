import fitz  # PyMuPDF
import os

def extract_pages_as_png(pdf_path, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        zoom = 2  
        mat = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=mat, alpha=False)

        output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(output_path)
        print(f"Saved {output_path}")

    print(f"All pages extracted and saved to {output_folder}")

if __name__ == "__main__":
    pdf_path = "test1.pdf"  
    output_folder = "output_images"  
    extract_pages_as_png(pdf_path, output_folder)
