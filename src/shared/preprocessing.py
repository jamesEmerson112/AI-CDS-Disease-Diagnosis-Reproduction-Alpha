"""
Text preprocessing utilities for symptom and diagnosis text.
"""

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Initialize stop words (downloaded on first use)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))


def preprocess_sentence(text):
    """
    Preprocess a sentence for embedding generation.
    - Handles punctuation spacing
    - Lowercases text
    - Removes stopwords and punctuation
    """
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)


def preprocess_diagnosis(diagnosis):
    """
    Preprocess diagnosis text by removing DRG type prefixes and normalizing.
    """
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.rstrip()
    diagnosis_list = diagnosis.split('--')
    diagnosis_list = list(set(diagnosis_list))
    diagnosis_no_drgtype = list()

    for d in diagnosis_list:
        d = d.replace("apr:", "")
        d = d.replace("hcfa:", "")
        d = d.replace("ms:", "")
        diagnosis_no_drgtype.append(d)

    diagnosis_no_drgtype = list(set(diagnosis_no_drgtype))
    diagnosis_final = list()

    for x in diagnosis_no_drgtype:
        prefix = ''
        for d in diagnosis_list:
            if x in d:
                prefix = prefix + d[0:d.index(':')] + ","
        prefix = prefix[:-1]
        x = prefix + ":" + x
        diagnosis_final.append(x.rstrip())

    return diagnosis_final
