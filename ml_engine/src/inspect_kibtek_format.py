import os
from pypdf import PdfReader

def inspect_specific_files():
    download_dir = "c:/Users/User/Downloads"
    files_to_inspect = [
        "Bölgesel Tüketici Adetleri 2023.pdf",
        "2024 statistic.pdf",
        "2023 TUKETIM RAPORU.pdf",
        "2023 KESINTI RAPORU.pdf"
    ]
    
    output_log = "c:/Users/User/Downloads/scratch-main/ml_engine/src/kibtek_inspected.txt"
    with open(output_log, "w", encoding="utf-8") as out:
        for filename in files_to_inspect:
            path = os.path.join(download_dir, filename)
            if os.path.exists(path):
                out.write(f"\n================ INSPECTING: {filename} ================\n")
                reader = PdfReader(path)
                out.write(f"Total Pages: {len(reader.pages)}\n")
                
                # Print first 2 pages
                for idx in range(min(2, len(reader.pages))):
                    out.write(f"\n--- Page {idx+1} ---\n")
                    text = reader.pages[idx].extract_text()
                    out.write(text[:1500])
                    out.write("\n")
                out.write("========================================================\n\n")
            else:
                out.write(f"[WARNING] File not found: {filename}\n")
    print(f"Inspection complete. Written to: {output_log}")

if __name__ == "__main__":
    inspect_specific_files()
