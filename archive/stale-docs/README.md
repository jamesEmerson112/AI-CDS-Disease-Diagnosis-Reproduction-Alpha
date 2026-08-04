# AI-CDS-Disease-Diagnosis-Reproduction

**Reproduction of:** *AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity*

An AI-powered clinical decision support system for disease diagnosis using sentence embeddings. This system predicts diagnoses based on symptom similarity using sent2vec embeddings and K-fold cross-validation.

> **Note:** This repository reproduces the experiments from the paper *"AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* by Comito et al. (2022), published in IEEE Access. The original authors proposed an innovative Clinical Decision Support (CDS) framework leveraging semantic similarity between patient symptoms and preliminary diagnoses through Natural Language Processing (NLP). By integrating electronic health records (EHR), semantic embeddings (Sent2Vec), and hybrid AI methodologies, the authors demonstrated improved accuracy and clinical interpretability in predicting discharge diagnoses for hospitalized patients.

## Overview

This project implements a machine learning approach to disease diagnosis by:
- Converting symptoms and diagnoses to vector embeddings using sent2vec
- Computing similarity between test symptoms and training data
- Predicting diagnoses based on highest similarity scores
- Evaluating performance using 10-fold cross-validation

## Quick Start

**For the impatient:** Get up and running in 5 minutes with conda:

```bash
# 1. Clone the repository
git clone https://github.com/jamesEmerson112/AI-CDS-Disease-Diagnosis-Reproduction.git
cd AI-CDS-Disease-Diagnosis-Reproduction

# 2. Create conda environment (installs Python 3.9 + dependencies)
conda env create -f environment.yml

# 3. Activate environment
conda activate disease-diagnosis

# 4. Verify setup
python test_setup.py

# 5. Run the project (if all tests pass)
python CS2V.py
```

**Note:** The conda setup automatically handles path detection in `utils/Constants.py`. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for team collaboration details.

**⚠️ Known Issue:** The `sent2vec` dependency may fail to install on Windows due to compiler requirements. The conda environment will still be created with all other dependencies. See detailed installation instructions below.

---

## Prerequisites

- **Python**: 3.8 or 3.9 (recommended)
  - **⚠️ Important:** The pre-compiled `util_cy.c` file has compatibility issues with Python 3.10+
  - For Python 3.10+, you'll need the original `.pyx` source file to recompile
- **C Compiler**: Required for Cython compilation (GCC on Linux/Mac, MSVC on Windows)
- **Pre-trained Model**: `BioSentVec_PubMed_MIMICIII-bigram_d700.bin` (~21GB)
  - This specific biomedical sentence embedding model is required
  - Download from NCBI BioSentVec or biomedical NLP resources

### Required Python Packages
- numpy
- scipy
- scikit-learn
- gensim
- sent2vec
- Cython
- nltk (with stopwords and punkt_tab data)
- matplotlib (for timing report PDF generation)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jamesEmerson112/AI-CDS-Disease-Diagnosis-Reproduction.git
cd AI-CDS-Disease-Diagnosis-Reproduction
```

### 2. Verify Python Version
Check that you have Python 3.8 or 3.9 installed:
```bash
python --version
```

**⚠️ Important:** Must be Python 3.8.x or 3.9.x (NOT 3.10 or higher due to Cython compatibility)

If you need Python 3.9, download from: https://www.python.org/downloads/

### 3. Create Virtual Environment (Recommended)
Create an isolated environment for the project dependencies:

```bash
python -m venv venv
```

**Activate the virtual environment:**

On Windows:
```bash
venv\Scripts\activate
```

On Linux/Mac:
```bash
source venv/bin/activate
```

**Verify activation:** You should see `(venv)` at the start of your terminal prompt.

### 4. Install Python Dependencies
Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```

This installs:
- numpy >= 1.19.0
- scipy >= 1.5.0
- scikit-learn >= 0.23.0
- gensim >= 3.8.0
- sent2vec >= 0.1.3
- Cython >= 0.29.0

### 5. Compile Cython Module
The `util_cy.c` file contains Cython-optimized methods. Compile it before running:
```bash
python setup.py build_ext --inplace
```

**Note:** If compilation fails with Python 3.10+, you'll need to use Python 3.8 or 3.9, or obtain the original `.pyx` source file from the authors.

### 6. NLTK Data (Automatic Download)

**Good news:** NLTK data packages are **automatically downloaded** when needed! 

The project requires NLTK's stopwords and punkt_tab data packages for text processing. Both `CS2V.py` and `test_setup.py` automatically detect missing NLTK data and download it on first run.

**What happens on first run:**
```
[INFO] First-time setup: Downloading required NLTK data packages...
[INFO] Missing packages: stopwords, punkt_tab
[INFO] Downloading 'stopwords'... ✓
[INFO] Downloading 'punkt_tab'... ✓
[SUCCESS] NLTK data download complete!
```

**Storage location:**
- Linux/Mac: `~/nltk_data/`
- Windows: `C:\Users\<username>\nltk_data\`

**Manual download (only if automatic fails):**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 7. Download Pre-trained Model
Download the required BioSentVec model:

**Required Model:**
- **Filename:** `BioSentVec_PubMed_MIMICIII-bigram_d700.bin`
- **Size:** ~21GB
- **Location:** Place in the project root directory

**Download Sources:**
- NCBI BioSentVec: [https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/)
- Original paper supplementary materials
- Biomedical NLP model repositories

The code expects this exact filename and will look for it in the current working directory.

## Setup Verification

### Automated Setup Testing

The project includes `test_setup.py` - an automated verification script that checks your installation.

**What it does:**
1. **Constants Import**: Tests if utils.Constants can be imported and CH_DIR is detected
2. **Path Validation**: Verifies the project directory exists
3. **File Check**: Confirms all key project files are present (CS2V.py, Dataset/, etc.)
4. **Dataset Structure**: Validates 10 cross-validation folds exist
5. **Dependencies**: Checks that all required Python packages are installed

**How to run:**
```bash
python test_setup.py
```

**Expected output:**
```
Testing Disease Diagnosis Project Setup

[Test 1] Importing Constants...
✅ SUCCESS: Constants imported
   Project path detected: /path/to/project

[Test 2] Verifying project path exists...
✅ SUCCESS: Path exists

[Test 3] Checking key project files...
   ✅ Found: CS2V.py
   ✅ Found: Symptoms-Diagnosis.txt
   ...
✅ SUCCESS: All key files found

[Test 4] Checking Dataset directory...
✅ SUCCESS: Found 10 folds in Dataset/

[Test 5] Checking Python dependencies...
   ✅ numpy
   ✅ scipy
   ✅ scikit-learn
   ✅ gensim
   ✅ Cython
✅ SUCCESS: All dependencies installed

🎉 ALL TESTS PASSED!
```

**When to run:**
- After initial setup
- Before running CS2V.py
- After switching Python environments
- When troubleshooting issues

### Manual Verification (Optional)

You can also verify components manually:

```bash
# Check Python version (should be 3.8.x or 3.9.x)
python --version

# Verify Cython module compiled successfully
ls -lh util_cy*.so  # On Linux/Mac
# or
dir util_cy*.pyd    # On Windows

# Confirm model file exists
ls -lh BioSentVec_PubMed_MIMICIII-bigram_d700.bin

# Test imports
python -c "import util_cy; print('Cython module loaded successfully')"
```

## Configuration

### Important: Update Working Directory
Before running the project, you **must** update the `CH_DIR` constant in `utils/Constants.py`:

```python
CH_DIR = "/path/to/your/AI-CDS-Disease-Diagnosis-Reproduction"
```

Replace with your actual project directory path.

### Configurable Parameters (utils/Constants.py)

- `K_FOLD`: Number of cross-validation folds (default: 10)
- `TOP_K_LOWER_BOUND`: Minimum Top-K value (default: 10)
- `TOP_K_UPPER_BOUND`: Maximum Top-K value (default: 60)
- `TOP_K_INCR`: Top-K increment step (default: 10)
- `PRUNING_SIMILARITY`: Similarity threshold for pruning (default: 0.5)
- `MIN_SIMILARITY`: Minimum similarity threshold (default: 0)

## How to Run

### Basic Execution
```bash
python CS2V.py
```

The script will:
1. Load the symptoms-diagnosis dataset
2. Load the pre-trained embedding model
3. Generate embeddings for symptoms and diagnoses
4. Run 10-fold cross-validation
5. Compute predictions and performance metrics
6. Generate output files with results

### Expected Output
- **Console**: Progress information for each fold
- **Files**:
  - `Prediction Output_[timestamp]/` - Prediction results for each fold
  - `Prediction Symptom Details_[timestamp]/` - Detailed symptom-level analysis
  - `PerformanceIndex.txt` - Aggregated performance metrics

### Runtime
Processing time depends on:
- Dataset size
- Number of folds
- Model complexity
- Hardware specifications

Expect several minutes to hours for complete execution.

## Dataset Structure

### Input Format
**Symptoms-Diagnosis.txt**: Semicolon-separated file with format:
```
HADM_ID;SUBJECT_ID;ADMIT_TIME;DISCHARGE_TIME;SYMPTOMS;DIAGNOSIS
```

### Cross-Validation Folds
The `Dataset/` directory contains 10 pre-split folds (Fold0-Fold9):
- `TrainingSet.txt`: Training data for each fold
- `TestSet.txt`: Test data for each fold

Each fold represents a different train-test split for cross-validation.

## Output Description

### Prediction Output
- **Prediction files**: Individual prediction results for each test case
- **Performance metrics**: Confusion matrices, accuracy, precision, recall, F-score
- **Similarity matrices**: Computed similarities between test and training samples

### Performance Index
The system evaluates predictions using:
- **TP** (True Positives)
- **FP** (False Positives)
- **P** (Precision)
- **R** (Recall)
- **FS** (F-Score)
- **PR** (Performance Ratio)

Results are computed for:
- Maximum similarity predictions
- Top-K similarity predictions (K = 10, 20, 30, 40, 50)

## Project Structure

```
AI-CDS-Disease-Diagnosis-Reproduction/
├── CS2V.py                      # Main execution script
├── test_setup.py                # Automated setup verification script
├── setup.py                     # Cython build configuration
├── util_cy.c                    # Cython-optimized utility methods
├── Symptoms-Diagnosis.txt       # Sample input data
├── README.md                    # Main documentation (this file)
├── SETUP_GUIDE.md               # Quick-start guide for teams
├── FILE_DESCRIPTIONS.txt        # Brief file descriptions (from original)
├── environment.yml              # Conda environment specification
├── requirements.txt             # Pip dependencies
├── entity/                      # Data model classes
│   ├── __init__.py
│   ├── Admission.py            # Admission data model
│   ├── Drgcodes.py             # DRG codes model
│   ├── Symptom.py              # Symptom model
│   └── SymptomsDiagnosis.py    # Main symptoms-diagnosis model
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── Constants.py            # Configuration constants (auto-detects path)
└── Dataset/                     # Cross-validation datasets
    ├── Fold0/
    │   ├── TrainingSet.txt
    │   └── TestSet.txt
    ├── Fold1/ ... Fold9/        # Additional folds
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'util_cy'**
- Ensure Cython module is compiled: `python setup.py build_ext --inplace`
- Check if `util_cy.so` (Linux/Mac) or `util_cy.pyd` (Windows) exists
- Verify you're using Python 3.8 or 3.9

**Cython Compilation Errors**
- If using Python 3.10+: Switch to Python 3.8 or 3.9
- The pre-compiled `util_cy.c` uses deprecated Python C API features
- Alternative: Contact original authors for `.pyx` source to recompile

**FileNotFoundError**
- Verify `CH_DIR` in `utils/Constants.py` points to correct directory
- Ensure all dataset files exist in `Dataset/Fold*/`

**Model Loading Error**
- Confirm model file exists: `BioSentVec_PubMed_MIMICIII-bigram_d700.bin`
- Must be placed in project root directory
- File size should be ~21GB
- Verify model format is compatible with sent2vec/gensim

**Memory Error**
- Minimum 8GB RAM recommended, 16GB+ ideal
- Close other applications during processing
- Consider using cloud computing (Google Colab, Kaggle) for 12-16GB RAM
- Reduce dataset size if necessary
- Process fewer folds at a time

## Citation

This project reproduces the work from the following paper:

**Paper Citation:**

> Comito, C., Falcone, D., & Forestiero, A. (2022). *AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity*. IEEE Access, 10, 6878–6888.
> DOI: [10.1109/ACCESS.2022.3142100](https://doi.org/10.1109/ACCESS.2022.3142100)

**BibTeX:**
```bibtex
@article{comito2022ai,
  title={AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity},
  author={Comito, Carmela and Falcone, Deborah and Forestiero, Agostino},
  journal={IEEE Access},
  volume={10},
  pages={6878--6888},
  year={2022},
  publisher={IEEE},
  doi={10.1109/ACCESS.2022.3142100}
}
```

**Original Resources:**
- Original codebase: [http://staff.icar.cnr.it/diseaseDiagnosis.zip](http://staff.icar.cnr.it/diseaseDiagnosis.zip)
- Dataset: MIMIC-III v1.4 ([PhysioNet](https://physionet.org/content/mimiciii/))

If you use this reproduction or the original work in your research, please cite the paper above.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open an issue on GitHub.
