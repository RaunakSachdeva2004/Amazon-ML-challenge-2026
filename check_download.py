"""
Helper script to detect, unzip, and arrange dataset files when download finishes.
"""
import os
import glob
import zipfile
import shutil

DOWNLOAD_DIR = os.path.expanduser(r"~\Downloads")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Scanning {DOWNLOAD_DIR} for recently downloaded zip archives...")

zip_files = sorted(glob.glob(os.path.join(DOWNLOAD_DIR, "*.zip")), key=os.path.getmtime, reverse=True)

found_target = None
for zpath in zip_files:
    try:
        with zipfile.ZipFile(zpath, 'r') as zf:
            namelist = zf.namelist()
            if any('train_source1.tsv' in n or 'test_source1.tsv' in n for n in namelist):
                found_target = zpath
                print(f"FOUND DATASET ZIP: {zpath}")
                break
    except Exception as e:
        continue

if found_target:
    print(f"Extracting {found_target} into project workspace...")
    with zipfile.ZipFile(found_target, 'r') as zf:
        zf.extractall(PROJECT_DIR)
    print("Extraction complete!")
else:
    print("No matching dataset zip file detected in Downloads yet.")
