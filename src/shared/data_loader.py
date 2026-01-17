"""
Data loading utilities for cross-validation folds and timing.
"""

import os
from datetime import datetime

from src.shared.constants import CH_DIR, FMT
from src.shared.preprocessing import preprocess_sentence


def current_time():
    """Get current timestamp as formatted string."""
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def elapsed_time(start_time, end_time):
    """Calculate elapsed time between two timestamps."""
    return datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)


def load_dataset(nFold, name):
    """
    Load dataset from a specific fold.

    Args:
        nFold: Fold number (0-9)
        name: File name (TrainingSet.txt or TestSet.txt)

    Returns:
        List of dictionaries mapping patient ID to preprocessed symptoms
    """
    dataset_dir = os.path.join(CH_DIR, 'data', 'folds', 'Fold' + str(nFold)) + "/"
    dataset_name = dataset_dir + name
    dataset_file = open(dataset_name, "r")
    dataset = list()

    for line in dataset_file:
        index = line[0:line.index("_")]
        symptoms = line[line.index("_") + 1: len(line) - 1]
        symptoms_list = symptoms.split(',')
        symptoms_list_preproc = list()

        for s in symptoms_list:
            symptoms_list_preproc.append(preprocess_sentence(s))
        dataset.append({index: symptoms_list_preproc})

    return dataset
