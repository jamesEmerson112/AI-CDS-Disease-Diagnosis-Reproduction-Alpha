#CONSTANT

import os
from pathlib import Path

# Auto-detect project root (works on Windows, WSL, Linux, Mac)
# This file is in utils/, so parent.parent gets us to the project root
CH_DIR = str(Path(__file__).parent.parent.absolute())

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
