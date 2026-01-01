import os
import gensim
from gensim.models import KeyedVectors
from scipy import spatial
from sklearn.model_selection import train_test_split
import numpy as np

from datetime import datetime
from utils.Constants import *

import sent2vec
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

def cosine_similarity(u, v):
    """
    Compute cosine similarity between two vectors.
    Pure Python implementation replacing the Cython cdef function.
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


def predictS2V(i, index, test_admission, test_symptoms, x_train, nrow, ncol, embendings_symptoms, 
               embendings_diagnosis, admissions, similarity_matrix,
               confusion_matrix_mean_max, confusion_matrix_max, confusion_matrix_Top_K_mean_max_dict, 
               confusion_matrix_Top_K_max_dict, directory_prediction,
               directory_prediction_details, performance_out_file):
    
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


def current_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def elapsed_time(start_time, end_time):
    return datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)


def load_dataset(nFold, name):
    dataset_dir = os.getcwd() + '/Dataset/Fold' + str(nFold) + "/"
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


def embending_symptoms(model, admissions):
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


def load_model():
    start_time = current_time()
    print("LOAD MODEL Start time: ", start_time)
    model = sent2vec.Sent2vecModel()
    model.load_model(os.getcwd() + '/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
    
    end_time = current_time()
    print("LOAD MODEL End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("LOAD MODEL Execution time: " + str(time_diff))
    
    return model


def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()
    
    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
    
    return ' '.join(tokens)


def preprocess_diagnosis(diagnosis):
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


def get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, method):
    MIN_SIMILARITY = 0
    max_diagnosis_similarity = dict()
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
    MIN_SIMILARITY = 0
    max_diagnosis_similarity = dict()
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
    similarity = 0
    
    for c in gt_diagnosis:
        if c in predicted_diagnosis:
            similarity += 1
        else:
            similarity += 0
    
    return similarity / len(gt_diagnosis)


def get_diagnosis_similarity_by_drgcode(gt_diagnosis, predicted_diagnosis):
    for c in gt_diagnosis:
        if c in predicted_diagnosis:
            return 1
    return 0


# Confusion Matrix
# { X : [TP, FP] }
def init_confusion_matrix():
    confusion_matrix = dict()
    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        confusion_matrix.update({i: [0 for x in range(2)]})
    return confusion_matrix


# Performance Matrix
# { X : [TP, FP, P, R, FS, PR] }
def init_performance_matrix():
    confusion_matrix = dict()
    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        confusion_matrix.update({i: [0 for x in range(6)]})
    return confusion_matrix


def containGreaterOrEqualsValue(topK, top_similarities, b):
    for i in range(0, topK):
        if i < len(top_similarities) and top_similarities[i] >= b:
            return True
    return False


def compute_performance_index(confusion_matrix, nrow, performance_out_file):
    performance_out_file.write(PERFORMANCE_INDEX_HEADER)
    
    for cm in {1, 0.9, 0.8, 0.7, 0.6}:
        values = confusion_matrix.get(cm)
        tp = values[TP]
        fp = values[FP]
        
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        recall = tp / nrow
        
        if recall + precision != 0:
            f_score = (2 * recall * precision) / (recall + precision)
        else:
            f_score = 0
        
        prediction_rate = (tp + fp) / nrow
        
        performance_out_file.write(
            str(cm) + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(
                f_score) + "\t" + str(prediction_rate) + "\n")
        performance_out_file.flush()


def compute_aggregated_performance_index(confusion_matrix, performance_matrix, nrow, performance_out_file):
    performance_out_file.write(PERFORMANCE_INDEX_HEADER)
    
    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        # UPDATE PERFORMANCE INDEX
        confusion_values = confusion_matrix.get(i)
        values = performance_matrix.get(i)
        tp = confusion_values[TP]
        fp = confusion_values[FP]
        
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        recall = tp / nrow
        
        if recall + precision != 0:
            f_score = (2 * recall * precision) / (recall + precision)
        else:
            f_score = 0
        
        prediction_rate = (tp + fp) / nrow
        
        performance_out_file.write(
            str(i) + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(
                f_score) + "\t" + str(prediction_rate) + "\n")
        performance_out_file.flush()
        
        # UPDATE PERFORMANCE INDEX
        values[TP] += tp
        values[FP] += fp
        values[P] += precision
        values[R] += recall
        values[FS] += f_score
        values[PR] += prediction_rate


def print_performance_index(performance_matrix, performance_out_file):
    performance_out_file.write(PERFORMANCE_INDEX_HEADER)
    
    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        values = performance_matrix.get(i)
        performance_out_file.write(
            str(i) + "\t" + str(values[TP] / K_FOLD) + "\t" + str(values[FP] / K_FOLD) + "\t" + str(
                values[P] / K_FOLD) + "\t" + str(values[R] / K_FOLD) +
            "\t" + str(values[FS] / K_FOLD) + "\t" + str(values[PR] / K_FOLD) + "\n")


def print_log(file, str, log):
    if log == LOG:
        file.write(str)
