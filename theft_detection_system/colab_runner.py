"""
Electricity Theft Detection System - Colab Runner
===================================================
Run this script in Google Colab (Pro+ A100 recommended) to execute the full pipeline.
"""

# =============================================================================
# [CELL 1] Setup & Clone Repository
# =============================================================================
import os
import sys

print("=========================================")
print(" 1. CLONING REPOSITORY & INSTALLING DEPS")
print("=========================================")

os.system('git clone https://github.com/ericdomah/scratch.git /content/scratch')
os.system('pip install -q -r /content/scratch/theft_detection_system/requirements.txt')

sys.path.insert(0, '/content/scratch/theft_detection_system')
os.chdir('/content/scratch/theft_detection_system')

print("Setup complete.")


# =============================================================================
# [CELL 2] Mount Google Drive & Check Data
# =============================================================================
print("\n=========================================")
print(" 2. MOUNTING GOOGLE DRIVE & DATA CHECK")
print("=========================================")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"Drive mount failed: {e}")

DATA_PATH = "data/data.csv"
if not os.path.exists(DATA_PATH):
    print(f"\n[WARNING] Data file not found at {DATA_PATH}")
    print("Please upload SGCC 'data.csv' to /content/scratch/theft_detection_system/data/")
    os.makedirs("data", exist_ok=True)
else:
    print(f"[OK] Data file found: {DATA_PATH}")


# =============================================================================
# [CELL 3] Run Baseline Machine Learning Models
# =============================================================================
print("\n=========================================")
print(" 3. RUNNING BASELINE ML MODELS")
print("=========================================")
os.system('python main.py --mode baseline')


# =============================================================================
# [CELL 4] Run Deep Learning Model (CNN-LSTM Hybrid)
# =============================================================================
print("\n=========================================")
print(" 4. RUNNING DEEP LEARNING (CNN-LSTM)")
print("=========================================")
os.system('python main.py --mode deep --model cnn_lstm')


# =============================================================================
# [CELL 5] Run Deep Learning Model (Transformer)
# =============================================================================
print("\n=========================================")
print(" 5. RUNNING DEEP LEARNING (TRANSFORMER)")
print("=========================================")
os.system('python main.py --mode deep --model transformer')


# =============================================================================
# [CELL 6] Run Explainable AI (SHAP & Permutation Importance)
# =============================================================================
print("\n=========================================")
print(" 6. RUNNING EXPLAINABLE AI")
print("=========================================")
os.system('python main.py --mode explain')


# =============================================================================
# [CELL 7] Backup Results to Google Drive
# =============================================================================
print("\n=========================================")
print(" 7. SAVING RESULTS TO GOOGLE DRIVE")
print("=========================================")
import shutil

DRIVE_DIR = "/content/drive/MyDrive/TheftDetection"
os.makedirs(DRIVE_DIR, exist_ok=True)

if os.path.exists("outputs"):
    print(f"Copying outputs to {DRIVE_DIR}...")
    shutil.copytree("outputs", f"{DRIVE_DIR}/outputs", dirs_exist_ok=True)
    print("Backup complete!")
else:
    print("No outputs directory found to backup.")
