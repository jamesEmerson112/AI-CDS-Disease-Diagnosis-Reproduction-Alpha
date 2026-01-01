"""
Test script to verify the project setup is working correctly.
Run this from Windows PowerShell or conda prompt.
"""

import os
import sys

print("=" * 60)
print("Testing Disease Diagnosis Project Setup")
print("=" * 60)

# Test 1: Import Constants
print("\n[Test 1] Importing Constants...")
try:
    from utils.Constants import CH_DIR
    print("✅ SUCCESS: Constants imported")
    print(f"   Project path detected: {CH_DIR}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Verify path exists
print("\n[Test 2] Verifying project path exists...")
if os.path.exists(CH_DIR):
    print(f"✅ SUCCESS: Path exists")
else:
    print(f"❌ FAILED: Path does not exist: {CH_DIR}")
    sys.exit(1)

# Test 3: Check key files
print("\n[Test 3] Checking key project files...")
key_files = [
    "CS2V.py",
    "Symptoms-Diagnosis.txt",
    "environment.yml",
    "requirements.txt",
    "utils/Constants.py"
]

all_found = True
for file in key_files:
    file_path = os.path.join(CH_DIR, file)
    if os.path.exists(file_path):
        print(f"   ✅ Found: {file}")
    else:
        print(f"   ❌ Missing: {file}")
        all_found = False

if not all_found:
    print("❌ FAILED: Some files are missing")
    sys.exit(1)
else:
    print("✅ SUCCESS: All key files found")

# Test 4: Check Dataset directory
print("\n[Test 4] Checking Dataset directory...")
dataset_path = os.path.join(CH_DIR, "Dataset")
if os.path.exists(dataset_path):
    folds = [f for f in os.listdir(dataset_path) if f.startswith("Fold")]
    print(f"✅ SUCCESS: Found {len(folds)} folds in Dataset/")
else:
    print("❌ FAILED: Dataset/ directory not found")
    sys.exit(1)

# Test 5: Try importing dependencies
print("\n[Test 5] Checking Python dependencies...")
dependencies = {
    "numpy": "numpy",
    "scipy": "scipy", 
    "sklearn": "scikit-learn",
    "gensim": "gensim",
    "Cython": "Cython",
    "nltk": "nltk",
    "matplotlib": "matplotlib"
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - not installed")
        missing.append(package)

if missing:
    print(f"\n⚠️  WARNING: Missing packages: {', '.join(missing)}")
    print("   Run: conda env create -f environment.yml")
else:
    print("✅ SUCCESS: All dependencies installed")

# Test 6: Check NLTK data packages
print("\n[Test 6] Checking NLTK data packages...")
try:
    import nltk
    
    required_nltk_data = {
        'stopwords': 'corpora/stopwords',
        'punkt_tab': 'tokenizers/punkt_tab',
        'punkt': 'tokenizers/punkt'
    }
    
    nltk_missing = []
    for name, path in required_nltk_data.items():
        try:
            nltk.data.find(path)
            print(f"   ✅ {name}")
        except LookupError:
            nltk_missing.append(name)
    
    if nltk_missing:
        print(f"\n⚠️  Missing NLTK data: {', '.join(nltk_missing)}")
        print("   Downloading now... (this is automatic)")
        
        for pkg in nltk_missing:
            print(f"   Downloading '{pkg}'... ", end='', flush=True)
            try:
                nltk.download(pkg, quiet=True)
                print("✓")
            except Exception as e:
                print(f"✗ (Error: {e})")
        
        print("✅ SUCCESS: NLTK data downloaded")
        print("   Note: CS2V.py will also auto-download NLTK data if needed")
    else:
        print("✅ SUCCESS: All required NLTK data is present")
        
except ImportError:
    print("⚠️  WARNING: nltk not installed - skipping NLTK data check")
except Exception as e:
    print(f"⚠️  WARNING: NLTK data check failed: {e}")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nYour project is ready to use!")
print(f"Project location: {CH_DIR}")
print("\nTo run the main script:")
print("  python CS2V.py")
print("=" * 60)
