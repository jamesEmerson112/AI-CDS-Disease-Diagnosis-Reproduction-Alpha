"""
Similarity computation utilities - cosine similarity and diagnosis comparison.
"""

import numpy as np
from scipy import spatial

from src.shared.preprocessing import preprocess_sentence


def cosine_similarity(u, v):
    """
    Compute cosine similarity between two vectors.
    Pure Python implementation replacing the Cython cdef function.

    Returns value between -1 and 1 (typically 0-1 for text embeddings).
    """
    assert len(u) == len(v)

    uv = 0.0
    uu = 0.0
    vv = 0.0

    for i in range(len(u)):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]

    cos_theta = 0.0
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)

    return cos_theta


def get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, method):
    """
    Get maximum similarity between ground truth and predicted diagnoses using pre-computed embeddings.
    """
    MIN_SIMILARITY = 0
    max_similarity = MIN_SIMILARITY

    for x in gt_diagnosis:
        x_description = x[x.index(':') + 1:len(x)]
        for y in predicted_diagnosis:
            y_description = y[y.index(':') + 1:len(y)]
            emb_diagnosis_to_predict = embendings_diagnosis.get(x_description)
            emb_diagnosis_predicted = embendings_diagnosis.get(y_description)
            diagnosis_similarity = cosine_similarity(emb_diagnosis_to_predict[0], emb_diagnosis_predicted[0])

            if diagnosis_similarity > max_similarity:
                max_similarity = diagnosis_similarity

    return max_similarity


def get_diagnosis_similarity_by_description_max_model(model, gt_diagnosis, predicted_diagnosis, method):
    """
    Get maximum similarity between diagnoses using model to generate embeddings on-the-fly.
    """
    MIN_SIMILARITY = 0
    max_similarity = MIN_SIMILARITY

    for x in gt_diagnosis:
        x_description = x[x.index(':') + 1:len(x)]
        for y in predicted_diagnosis:
            y_description = y[y.index(':') + 1:len(y)]
            emb_diagnosis_to_predict = model.embed_sentence(preprocess_sentence(x_description))
            emb_diagnosis_predicted = model.embed_sentence(preprocess_sentence(y_description))
            diagnosis_similarity = 1 - spatial.distance.cdist(emb_diagnosis_to_predict, emb_diagnosis_predicted, method)[0]

            if diagnosis_similarity > max_similarity:
                max_similarity = diagnosis_similarity

    return max_similarity[0]


def get_diagnosis_similarity_baseline(gt_diagnosis, predicted_diagnosis):
    """
    Simple baseline similarity - count exact matches.
    """
    similarity = 0

    for c in gt_diagnosis:
        if c in predicted_diagnosis:
            similarity += 1
        else:
            similarity += 0

    return similarity / len(gt_diagnosis)


def get_diagnosis_similarity_by_drgcode(gt_diagnosis, predicted_diagnosis):
    """
    Check if any diagnosis code matches.
    """
    for c in gt_diagnosis:
        if c in predicted_diagnosis:
            return 1
    return 0
