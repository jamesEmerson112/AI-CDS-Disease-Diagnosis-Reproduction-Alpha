# AI-CDS Disease Diagnosis System

Clinical Decision Support System for disease diagnosis prediction using patient symptom similarity.

Reproduces research from: *"AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (Comito et al., 2022)

## Quick Start

```bash
# Install dependencies
pip install -r config/requirements.txt

# Run baseline (Sent2Vec)
python scripts/run_baseline.py

# Run BERT models
pip install -r config/requirements_bert.txt
python scripts/run_bert_analysis.py
```

## Project Structure

```
src/                 # Source code
  models/            # Baseline and BERT implementations
  entity/            # Data classes
  utils/             # Utilities and constants
  evaluation/        # Evaluation modules
scripts/             # Entry point scripts
tests/               # Test files
data/                # Data files
  folds/             # 10-fold cross-validation data
  raw/               # Raw data files
  models/            # Pre-trained model files
output/              # Experiment outputs
docs/                # Documentation
config/              # Configuration files
```

## Documentation

See [docs/README.md](docs/README.md) for detailed documentation.
