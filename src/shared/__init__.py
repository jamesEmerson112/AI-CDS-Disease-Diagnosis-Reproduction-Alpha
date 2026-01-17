# Shared components for BioSentVec and BERT projects
# Import constants directly (no external dependencies)
from .constants import *

# Other imports are available but not auto-imported to avoid dependency issues
# Use explicit imports:
#   from src.shared.preprocessing import preprocess_sentence
#   from src.shared.similarity import cosine_similarity
#   from src.shared.data_loader import load_dataset
#   from src.shared.metrics import init_confusion_matrix
