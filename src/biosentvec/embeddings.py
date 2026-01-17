"""
BioSentVec-specific embedding generation functions.
"""

import os
import numpy as np
import sent2vec

from src.shared.constants import (
    CH_DIR, MIN_SIMILARITY, PRUNING_SIMILARITY,
    TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR, TP, FP
)
from src.shared.preprocessing import preprocess_sentence
from src.shared.similarity import cosine_similarity, get_diagnosis_similarity_by_description_max
from src.shared.data_loader import current_time, elapsed_time
from src.shared.metrics import compute_performance_index


def load_model():
    """
    Load the BioSentVec model from the data/models directory.
    """
    start_time = current_time()
    print("LOAD MODEL Start time: ", start_time)

    model = sent2vec.Sent2vecModel()
    # Try data/models first, then root directory for backwards compatibility
    model_path = os.path.join(CH_DIR, 'data', 'models', 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
    if not os.path.exists(model_path):
        model_path = os.path.join(CH_DIR, 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
    model.load_model(model_path)

    end_time = current_time()
    print("LOAD MODEL End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("LOAD MODEL Execution time: " + str(time_diff))

    return model


def embending_symptoms(model, admissions):
    """
    Generate embeddings for all symptoms in the admissions dataset.

    Returns dict mapping preprocessed symptom text to embedding vector.
    """
    start_time = current_time()
    print("EMBENDING SYMPTOMS Start time: ", start_time)
    embending_symptoms_dict = dict()

    for key in admissions:
        a = admissions.get(key)
        symptoms_list = a.symptoms.split(',')
        for s in symptoms_list:
            symptoms = preprocess_sentence(s)
            embs = model.embed_sentence(symptoms)
            embending_symptoms_dict.update({symptoms: embs})

    end_time = current_time()
    print("EMBENDING SYMPTOMS End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("EMBENDING SYMPTOMS Execution time: " + str(time_diff))

    return embending_symptoms_dict


def embending_diagnosis(model, admissions):
    """
    Generate embeddings for all diagnoses in the admissions dataset.

    Returns dict mapping diagnosis description to embedding vector.
    """
    start_time = current_time()
    print("EMBENDING DIAGNOSIS Start time: ", start_time)
    embending_diagnosis_dict = dict()

    for key in admissions:
        a = admissions.get(key)
        diagnosis_list = a.diagnosis
        for d in diagnosis_list:
            diagnosis_description = d[d.index(':') + 1:len(d)]
            embs = model.embed_sentence(preprocess_sentence(diagnosis_description))
            embending_diagnosis_dict.update({diagnosis_description: embs})

    end_time = current_time()
    print("EMBENDING DIAGNOSIS End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("EMBENDING DIAGNOSIS Execution time: " + str(time_diff))

    return embending_diagnosis_dict


def predictS2V(i, index, test_admission, test_symptoms, x_train, nrow, ncol, embendings_symptoms,
               embendings_diagnosis, admissions, similarity_matrix,
               confusion_matrix_mean_max, confusion_matrix_max, confusion_matrix_Top_K_mean_max_dict,
               confusion_matrix_Top_K_max_dict, directory_prediction,
               directory_prediction_details, performance_out_file):
    """
    Make prediction for a single test case using BioSentVec embeddings.
    """
    # OPEN PREDICTION OUTPUT FILE
    prediction_out_file = open(directory_prediction + test_admission.hadm_id + '.txt', 'w')
    detailed_out_file = open(directory_prediction_details + test_admission.hadm_id + '.txt', 'w')

    # START SINGLE PREDICTION TIME
    start_time = current_time()

    for j in range(0, ncol):
        # TRAIN SYMPTOMS
        train_symptoms = list(x_train[j].values())[0]
        max_symptoms_similarity = dict()

        # COMPUTE MAX SIMILARITY FOR EACH PAIR OF SYMPTOMS
        for x in test_symptoms:
            test_emb = embendings_symptoms.get(x)
            max_similarity = MIN_SIMILARITY
            max_symptom = None

            for y in range(0, len(train_symptoms)):
                train_emb = embendings_symptoms.get(train_symptoms[y])
                similarity = cosine_similarity(test_emb[0], train_emb[0])

                if similarity > max_similarity:
                    max_similarity = similarity
                    max_symptom = train_symptoms[y]

            if max_symptom is not None:
                max_symptoms_similarity.update({max_symptom + " for " + x: max_similarity})
            else:
                max_symptoms_similarity.update({"No Similar sympthom for " + x: max_similarity})

        # COMPUTE MEAN SIMILARITY
        min_den = len(test_symptoms)
        max_den = len(train_symptoms)
        if min_den > max_den:
            max_den = min_den
            min_den = len(train_symptoms)

        mean = 0
        for key in max_symptoms_similarity:
            mean = mean + float(max_symptoms_similarity.get(key))
        mean = mean / max_den
        similarity_matrix[i, j] = mean

    ###################################################################################
    # MAX SIMILARITY
    ###################################################################################
    max_val = MIN_SIMILARITY
    max_index = -1

    for j in range(0, ncol):
        if similarity_matrix[i, j] >= PRUNING_SIMILARITY and similarity_matrix[i, j] > max_val:
            max_val = similarity_matrix[i, j]
            max_index = j

    gt_diagnosis = test_admission.diagnosis

    if max_index != -1:
        most_similar_index = list(x_train[max_index].keys())[0]
        most_similar_symptoms = list(x_train[max_index].values())[0]
        most_similar_admission = admissions.get(most_similar_index)
        predicted_diagnosis = most_similar_admission.diagnosis
        diagnosis_similarity_max = get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, 'cosine')

        # UPDATE CONFUSION MATRIX
        for b in {1, 0.9, 0.8, 0.7, 0.6}:
            values = confusion_matrix_max.get(b)
            if diagnosis_similarity_max >= b:
                values[TP] += 1
            else:
                values[FP] += 1

    ###################################################################################
    # TOP-K SIMILARITY
    ###################################################################################
    similarity_array = np.array(similarity_matrix[i,])
    largest_indices = np.argsort(-1 * similarity_array)[:(TOP_K_UPPER_BOUND - TOP_K_INCR)]
    top_k_admission = list()
    top_similarities_max = list()
    index_top = 0

    for top_index in largest_indices:
        if similarity_matrix[i, int(top_index)] >= PRUNING_SIMILARITY:
            most_similar_index = list(x_train[top_index].keys())[0]
            most_similar_symptoms = list(x_train[top_index].values())[0]
            most_similar_admission = admissions.get(most_similar_index)
            predicted_diagnosis = most_similar_admission.diagnosis
            diagnosis_similarity_max = get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, 'cosine')
            top_similarities_max.append(diagnosis_similarity_max)
            index_top += 1

    # UPDATE CONFUSION MATRIX TOP K
    from src.shared.metrics import containGreaterOrEqualsValue
    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
        for b in {1, 0.9, 0.8, 0.7, 0.6}:
            values = confusion_matrix_Top_K_max.get(b)
            if containGreaterOrEqualsValue(topk, top_similarities_max, b):
                values[TP] += 1
            else:
                if len(top_similarities_max) > 0:
                    values[FP] += 1

    end_time = current_time()
    time_diff = elapsed_time(start_time, end_time)

    # CLOSE PREDICTION OUTPUT FILE
    prediction_out_file.close()
    detailed_out_file.close()

    # COMPUTE PERFORMANCE INDEX
    performance_out_file.write(str(i) + " - HADM_ID=" + str(test_admission.hadm_id) + ": PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
    compute_performance_index(confusion_matrix_max, nrow, performance_out_file)

    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
        performance_out_file.write(str(i) + " - HADM_ID=" + str(test_admission.hadm_id) + ": PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
        compute_performance_index(confusion_matrix_Top_K_max, nrow, performance_out_file)

    print("END PREDICTION " + test_admission.hadm_id + " in " + str(time_diff))
