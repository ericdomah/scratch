import sys
import re

def scrub_promotional_language(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Replacements based on examiner feedback
    replacements = [
        (r'(?i)is the first framework', "is, to the best of the author's knowledge, one of the few frameworks"),
        (r'(?i)super-hybrid', "integrated architecture"),
        (r'(?i)revolutionary', "proposed"),
        (r'(?i)intelligent engine', "experimental evaluation model"),
        (r'(?i)audit-grade', "rigorous"),
        (r'(?i)forensic intelligence', "contextual detection mechanism"),
        (r'(?i)cutting-edge', "proposed")
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Successfully scrubbed promotional language from thesis.")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    scrub_promotional_language("thesis/GridGuard_AI_Master_Thesis_Final.md")
