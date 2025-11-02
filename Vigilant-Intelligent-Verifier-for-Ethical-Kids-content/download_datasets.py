import os
import zipfile
import subprocess
import urllib.request
import shutil

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_DIR = "data"
DATASETS = {
    "NSFW_Model_GitHub": {
        "type": "git",
        "url": "https://github.com/GantMan/nsfw_model.git",
        "target": os.path.join(DATA_DIR, "nsfw_dataset")
    },
    "Hate_Speech_Text": {
        "type": "kaggle",
        "url": "hamzaboulahia/hate-speech-and-offensive-language",
        "target": os.path.join(DATA_DIR, "text_hate_speech")
    },
    "Reddit_Comments": {
        "type": "kaggle",
        "url": "ahmadjaved0975/reddit-comments-multilabel-classification",
        "target": os.path.join(DATA_DIR, "text_reddit")
    },
    "OpenNSFW2": {
        "type": "git",
        "url": "https://github.com/EBazarov/nsfw_data_scraper.git",
        "target": os.path.join(DATA_DIR, "opennsfw2")
    }
}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[+] Created directory: {path}")

def run_command(cmd):
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=False)

def unzip_file(filepath, dest_dir):
    print(f"[+] Extracting {filepath} ...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(filepath)

# ----------------------------
# STEP 1: CREATE BASE DIRS
# ----------------------------
print("=== Setting up dataset folders ===")
ensure_dir(DATA_DIR)
for subdir in ["images", "videos", "metadata"]:
    ensure_dir(os.path.join(DATA_DIR, subdir))

# ----------------------------
# STEP 2: DOWNLOAD DATASETS
# ----------------------------
print("\n=== Downloading datasets ===")

for name, info in DATASETS.items():
    print(f"\n--- {name} ---")
    target = info["target"]
    ensure_dir(target)

    if info["type"] == "git":
        if not os.path.exists(os.path.join(target, ".git")):
            run_command(f"git clone {info['url']} {target}")
        else:
            print(f"[✓] Repo already exists: {target}")

    elif info["type"] == "kaggle":
        # Ensure Kaggle is installed
        try:
            subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("[-] Kaggle CLI not found. Installing...")
            run_command("pip install kaggle")

        dataset_id = info["url"]
        zip_path = os.path.join(target, dataset_id.split('/')[-1] + ".zip")
        run_command(f'kaggle datasets download -d {dataset_id} -p {target}')

        # unzip downloaded dataset
        for f in os.listdir(target):
            if f.endswith(".zip"):
                unzip_file(os.path.join(target, f), target)

    else:
        print(f"[-] Unknown source type: {info['type']}")

# ----------------------------
# STEP 3: OPTIONAL - Download extra metadata (UCI)
# ----------------------------
print("\n=== Optional: Downloading UCI dataset metadata ===")

uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00337/"
uci_target = os.path.join(DATA_DIR, "uci_pornography_metadata")
ensure_dir(uci_target)
try:
    urllib.request.urlretrieve(uci_url, os.path.join(uci_target, "README.html"))
    print("[+] Saved UCI dataset info")
except Exception as e:
    print(f"[!] Could not fetch UCI dataset: {e}")

# ----------------------------
# STEP 4: ORGANIZE FILES
# ----------------------------
print("\n=== Organizing data folders ===")

# Move any text or CSV files into metadata/
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".csv") or file.endswith(".txt"):
            src = os.path.join(root, file)
            dst = os.path.join(DATA_DIR, "metadata", file)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"[+] Moved metadata file: {file}")

print("\n✅ All datasets downloaded and organized successfully!")

