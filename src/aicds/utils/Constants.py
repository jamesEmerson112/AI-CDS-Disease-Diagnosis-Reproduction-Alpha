#CONSTANT

import os
from pathlib import Path

# Auto-detect project root (works on Windows, WSL, Linux, Mac).
# This file is in src/aicds/utils/, so four parents get us to the project root:
# utils -> aicds -> src -> repo root. Every data path (data/raw, data/folds) and
# the BioSentVec model path hang off this, so an off-by-one here silently
# repoints the whole pipeline rather than raising.
CH_DIR = str(Path(__file__).parent.parent.parent.parent.absolute())

FMT = "%d/%m/%Y %H:%M:%S"
TRAIN = 'TrainingSet.txt'
TEST = 'TestSet.txt'
PERFORMANCE_INDEX_HEADER="\t TP \t FP \t  P \t R \t FS \t PR" + "\n"
PRUNING_SIMILARITY = 0.5
MIN_SIMILARITY = 0
TOP_K_LOWER_BOUND = 10
TOP_K_UPPER_BOUND = 60
TOP_K_INCR = 10
K_FOLD = 10
TP = 0
FP = 1
P = 2
R = 3
FS = 4
PR = 5
