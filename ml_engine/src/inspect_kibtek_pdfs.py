import os
import glob
import re

def inspect_downloads():
    download_dir = "c:/Users/User/Downloads"
    pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))
    print(f"Total PDFs found in Downloads: {len(pdf_files)}")
    
    # Filter files that might be KIB-TEK related
    kibtek_keywords = ["kibtek", "kib_tek", "audit", "report", "anomaly", "statistics", "istatistik"]
    kibtek_files = []
    
    for f in pdf_files:
        basename = os.path.basename(f).lower()
        if any(kw in basename for kw in kibtek_keywords):
            kibtek_files.append(f)
            
    print(f"KIB-TEK files identified: {len(kibtek_files)}")
    for f in kibtek_files[:10]:
        print(f" - {os.path.basename(f)}")
        
    # Read the first page of one file to inspect structure
    if kibtek_files:
        try:
            from pypdf import PdfReader
        except ImportError:
            import subprocess
            subprocess.check_call(["pip", "install", "pypdf"])
            from pypdf import PdfReader
            
        first_pdf = kibtek_files[0]
        print(f"\nInspecting contents of: {os.path.basename(first_pdf)}")
        reader = PdfReader(first_pdf)
        print(f"Number of pages: {len(reader.pages)}")
        if len(reader.pages) > 0:
            print("--- First 500 characters of Page 1 ---")
            print(reader.pages[0].extract_text()[:500])
            print("---------------------------------------")

if __name__ == "__main__":
    inspect_downloads()
