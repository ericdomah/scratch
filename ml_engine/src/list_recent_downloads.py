import os
import glob
import time

def list_recent_pdfs():
    download_dir = "c:/Users/User/Downloads"
    pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))
    
    # Sort files by modification time (most recent first)
    pdf_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Top 45 most recently modified/created PDF files in {download_dir}:")
    for idx, f in enumerate(pdf_files[:45]):
        mtime = time.ctime(os.path.getmtime(f))
        size_kb = os.path.getsize(f) / 1024
        print(f"{idx+1:02d}. Name: {os.path.basename(f)} | Size: {size_kb:.2f} KB | Modified: {mtime}")

if __name__ == "__main__":
    list_recent_pdfs()
