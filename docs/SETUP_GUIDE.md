# Disease Diagnosis Project - Setup Guide

## Quick Start (5 minutes)

### Prerequisites
- Windows with conda/Anaconda installed ✅ (You have this!)
- OR Linux/WSL with conda installed

### Setup Steps

#### 1. Clone/Open Project
```bash
cd C:\Users\voan2\Documents\GitHub\AI-CDS-Disease-Diagnosis-Reproduction
# Or on Linux/WSL:
# cd /path/to/AI-CDS-Disease-Diagnosis-Reproduction
```

#### 2. Create Conda Environment
```bash
conda env create -f environment.yml
```

This installs:
- Python 3.9
- numpy, scipy, scikit-learn, gensim, cython
- sent2vec

#### 3. Activate Environment
```bash
conda activate disease-diagnosis
```

#### 4. Verify Setup
```bash
python -c "from utils.Constants import CH_DIR; print('✅ Project path:', CH_DIR)"
```

#### 5. Run the Code
```bash
python CS2V.py
```

---

## Team Collaboration

### What Changed
- ✅ **No more manual path editing!** Constants.py auto-detects project location
- ✅ **One-command setup** with conda environment.yml
- ✅ **Works on Windows, Linux, Mac** - same commands everywhere

### For New Team Members
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Clone the repository
3. Run: `conda env create -f environment.yml`
4. Run: `conda activate disease-diagnosis`
5. Start working!

---

## Environment Management

### Activate environment (every time you work)
```bash
conda activate disease-diagnosis
```

### Deactivate when done
```bash
conda deactivate
```

### Update environment (if environment.yml changes)
```bash
conda env update -f environment.yml --prune
```

### Delete environment (if needed)
```bash
conda env remove -n disease-diagnosis
```

---

## Troubleshooting

### "conda: command not found"
- **Windows:** Open "Anaconda PowerShell Prompt" instead of regular PowerShell
- **Linux/Mac:** Run `conda init bash` then restart terminal

### "Module not found" errors
- Make sure environment is activated: `conda activate disease-diagnosis`
- Verify with: `conda env list` (should show * next to disease-diagnosis)

### Path issues
The project now auto-detects its location! If you see path errors:
1. Ensure you're in the project directory
2. Check that utils/Constants.py exists
3. Try: `python -c "from utils.Constants import CH_DIR; print(CH_DIR)"`

---

## What Was Fixed

### Before (Manual Setup Required)
```python
# In utils/Constants.py - everyone had to edit this!
CH_DIR = r"c:\Users\voan2\Documents\GitHub\AI-CDS-Disease-Diagnosis-Reproduction"
```

### After (Auto-Detect)
```python
# Now automatically detects project location
CH_DIR = str(Path(__file__).parent.parent.absolute())
```

**Result:** Clone and run - no configuration needed! ✨

---

## Files in This Repo

- `environment.yml` - Conda environment specification
- `requirements.txt` - Pip requirements (legacy, use environment.yml instead)
- `CS2V.py` - Main script
- `utils/Constants.py` - Auto-detecting configuration
- `Dataset/` - Training/test data (10-fold cross-validation)
- `BioSentVec_PubMed_MIMICIII-bigram_d700.bin` - 20.9GB model file

---

**Ready to go!** 🚀
