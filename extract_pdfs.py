import fitz  # PyMuPDF
import sys
import os

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

output_file = r"d:\TCS_AMD_BUILD_AI\extracted_text.txt"

files = [
    r"d:\TCS_AMD_BUILD_AI\TCSAMD AI Hackathon_Consolidated Usecases_200526_tracks.pdf",
    r"d:\TCS_AMD_BUILD_AI\TCS_AMD_Hackathon Handbook_ver_0.3.pdf",
]

with open(output_file, 'w', encoding='utf-8') as out:
    for f in files:
        out.write(f"\n{'='*80}\n")
        out.write(f"FILE: {f}\n")
        out.write('='*80 + '\n')
        try:
            doc = fitz.open(f)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                out.write(f"\n--- Page {page_num+1} ---\n")
                out.write(text)
            doc.close()
        except Exception as e:
            out.write(f"Error: {e}\n")

print(f"Text extracted to {output_file}")
