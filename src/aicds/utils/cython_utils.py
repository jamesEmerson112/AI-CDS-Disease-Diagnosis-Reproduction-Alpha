import os
import re
from scipy import spatial
import numpy as np

from datetime import datetime
from aicds.utils.Constants import *

# Imported AFTER the star import above so a future name added to Constants can
# never silently shadow LEGACY. aicds.config imports nothing from aicds, so
# there is no cycle.
from aicds.config import LEGACY

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# preprocess_sentence reads this at module scope, but for the whole history of
# this file it was never defined here -- eight separate call sites monkeypatched
# it in (four in src/scripts, four in tests), every one of them with the very
# same `set(stopwords.words('english'))`. Any new caller that forgot the
# incantation got a bare NameError from inside tokenization.
#
# Defining it here is numerically inert precisely BECAUSE all eight used an
# identical expression: the patches that remain still assign an equal value.
# They are removed in the very next commit, separately, because stop_words feeds
# tokenization -> embeddings -> every number in PerformanceIndex.txt, and a
# change that broad deserves an unambiguous culprit commit.
#
# The download guard matters for import order: bert_models imports this module
# before it calls ensure_nltk_data(), so on a machine without the corpus an
# eager set(stopwords.words(...)) would raise LookupError at import time.
try:
    stop_words = set(stopwords.words('english'))
except LookupError:  # corpus not downloaded yet
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))


def use_corrected_preprocessing(config):
    """Resolve ``config.preprocess_version`` to a boolean, or refuse.

    Deliberately NOT written as ``version != 'legacy'`` or
    ``version == 'corrected'``. Both of those silently pick a side when the
    value is a typo, and in this repository a silent side-pick is the exact
    failure mode the config seam exists to prevent: ``'correced'`` would run
    the legacy pipeline while the operator believed they had corrected it, and
    the resulting numbers would be filed under the wrong arm. Unknown values
    raise instead.

    Public, not ``_``-prefixed, because the BERT arm calls it too: FIX 4 in
    ``bert_models.compute_bert_diagnosis_embeddings`` gates on this same
    predicate. A guard against silently picking a side only works while every
    arm asks the same question, and a second, locally-written
    ``version == 'corrected'`` in the other module would reintroduce exactly
    the typo-swallowing this function exists to forbid.
    """
    version = config.preprocess_version
    if version == 'legacy':
        return False
    if version == 'corrected':
        return True
    raise ValueError(
        "unknown preprocess_version {!r}; expected 'legacy' or 'corrected'".format(version)
    )


# --------------------------------------------------------------------------
# FIX 3a -- 'w/o' collapsing to 'w' (docs/findings/06-preprocessing-defects.md
# defect 1). CORRECTED path only.
#
# 'w/o' is the ICD-9/DRG abbreviation for "without". preprocess_sentence pads
# '/' with spaces before tokenising, so 'w/o' becomes the three tokens
# ['w', '/', 'o']; '/' is dropped as punctuation and 'o' IS an NLTK English
# stopword, so all that survives is 'w' -- the abbreviation for *with*. Two
# clinically opposite DRG names end up byte-identical.
#
# Why rewrite to the full word rather than protect the literal 'w/o':
#   * 'without' is NOT in NLTK's English stopword list (verified: 'with' is,
#     'without' is not), and word_tokenize keeps it as a single token, so it
#     survives the existing filter with no change to that filter. Loosening
#     the stopword set instead would resurrect dozens of unrelated tokens and
#     make the corrected arm's delta impossible to attribute to negation.
#   * It is the expansion a sentence encoder can actually use. Both arms embed
#     this string; 'without' carries negation in the embedding space, whereas
#     a protected 'w/o' would be re-split into low-signal wordpieces anyway.
#
# Word boundaries are load-bearing. The committed dataset holds 52
# case-insensitive 'w/o' substrings, but one of them is 'w/oth' -- "with
# other", in 'Gall&bil cal w/oth w obs' -- which must not become 'withoutth'.
# \b on both sides matches exactly the 51 genuine occurrences and skips it.
#
# Deliberately out of scope, so the corrected arm's delta stays attributable:
# 'w/' (21 occurrences, "with"), 'd/t' ("due to"), and the loss of the
# stopword 'or' in "SEPTICEMIA OR SEVERE SEPSIS".
# --------------------------------------------------------------------------
_W_SLASH_O = re.compile(r'\bw/o\b', re.IGNORECASE)


def split_symptoms(text, config=LEGACY):
    """Split a SYMPTOMS field into its individual symptom strings.

    FIX 3b -- comma-split fragments (docs/findings/06-preprocessing-defects.md
    defect 2). This is the one place the split is defined; load_dataset,
    embending_symptoms and bert_models.compute_bert_symptom_embeddings all
    route through it, because three copies of the rule is three chances for
    the two arms to disagree about what a symptom *is*.

    LEGACY reproduces the original ``text.split(',')`` exactly -- same list,
    same objects, no reformatting -- so every legacy number is untouched.

    CORRECTED repairs the shredding. The SYMPTOMS field is ','-delimited but
    ICD-9 short titles contain commas of their own, so 'Pneumonia, organism
    NOS' arrives as 'Pneumonia' plus the orphan ' organism NOS'.

    Recovery is a HEURISTIC and it is INCOMPLETE. An earlier version of this
    docstring called it "exact rather than heuristic", on the grounds that
    every intra-label comma is followed by a space. **That is false** --
    adversarial verification found nine occurrences using a bare comma, and
    the claim was independently re-confirmed as wrong. Do not restore it.

    The rule -- a part beginning with a space is the tail of the part before
    it -- recovers 80 of the 89 shredded parts (1,805 tokens -> 1,725; 573
    unique strings -> 564). That is what stops 26 admissions sharing a
    meaningless ' organism NOS' token which scores a spurious 1.0 against each
    other under mean-of-max.

    KNOWN RESIDUAL, 9 occurrences across 4 labels, still shredded::

        4x 'stage NOS'   3x 'stage III'   1x 'uncomp'   1x 'pharyngoesoph'

    They belong to 'Pressure ulcer,stage NOS', 'Pressure ulcer,stage III',
    'Vascular dementia,uncomp' and 'Dysphagia,pharyngoesoph'. **Two are
    severity**, so a stage III pressure ulcer is currently indistinguishable
    from an unstaged one -- the same class of loss as the w/o negation bug.

    A second rule would close it: treat a part starting lower-case as a tail
    when the join still fits the CMS 24-character short-title cap. Measured,
    that yields 1,716 tokens / 561 unique and zero residual, firing on exactly
    those four labels. It is NOT implemented because it changes the contract
    of this function -- 'fever,cough' would stop splitting -- which breaks
    three existing tests that pin bare-comma separation. Landing it means
    deciding those tests are wrong and re-verifying, so it belongs in its own
    reviewed commit rather than riding along here.

    The check to re-run if the dataset ever changes: count parts beginning
    lower-case. It is 9 today and should never silently grow.

    A leading-space part with no predecessor cannot be a continuation, so it
    is kept as its own symptom rather than silently dropped.
    """
    parts = text.split(',')
    if not use_corrected_preprocessing(config):
        return parts

    rejoined = list()
    for part in parts:
        if rejoined and part.startswith(' '):
            rejoined[-1] = rejoined[-1] + ',' + part
        else:
            rejoined.append(part)
    return rejoined


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


def load_dataset(nFold, name, config=LEGACY):
    """Read one fold's train or test split.

    ``config.fold_dir`` selects which split to read. The default LEGACY value
    is ``'folds'``, so the resolved path is character-for-character what it was
    before the parameter existed -- every existing caller is numerically
    untouched. Pass ``CORRECTED`` (``'folds_grouped'``) for the
    leakage-free GroupKFold split.

    ``config`` is also forwarded to ``split_symptoms`` and
    ``preprocess_sentence``, so ``preprocess_version`` decides whether the
    comma-fragment rejoin (FIX 3b) and the ``w/o`` expansion (FIX 3a) apply to
    the symptom strings this returns. That forwarding must stay in step with
    whatever builds the symptom embedding dict -- the dict is keyed by
    preprocessed *text*, so a fold loaded under one version and embeddings
    built under the other simply fail to find each other.

    The slice below drops the last character of every line unconditionally,
    assuming a trailing newline. That is a real defect, but it is load-bearing:
    all 20 committed fold files end in 0x0a so it currently only eats the
    newline, and tests/test_characterize_dataset.py pins the behaviour in both
    directions. Do not "fix" it here -- it belongs behind a config field.
    """
    dataset_dir = os.path.join(CH_DIR, 'data', config.fold_dir, 'Fold' + str(nFold)) + "/"
    dataset_name = dataset_dir + name
    dataset_file = open(dataset_name, "r")
    dataset = list()
    
    for line in dataset_file:
        index = line[0:line.index("_")]
        symptoms = line[line.index("_") + 1: len(line) - 1]
        symptoms_list = split_symptoms(symptoms, config)
        symptoms_list_preproc = list()

        for s in symptoms_list:
            symptoms_list_preproc.append(preprocess_sentence(s, config))
        dataset.append({index: symptoms_list_preproc})
    
    return dataset


def embending_symptoms(model, admissions, config=LEGACY):
    """Build the baseline arm's symptom embedding dict, keyed by preprocessed text.

    ``config`` must match whatever ``load_dataset`` was called with: the keys
    here are the lookup keys there. Defaults to LEGACY so the untouched
    baseline path is unchanged.
    """
    start_time = current_time()
    print("EMBENDING SYMPTOMS Start time: ", start_time)
    embending_symptoms_dict = dict()

    for key in admissions:
        a = admissions.get(key)
        symptoms_list = split_symptoms(a.symptoms, config)
        for s in symptoms_list:
            symptoms = preprocess_sentence(s, config)
            embs = model.embed_sentence(symptoms)
            embending_symptoms_dict.update({symptoms: embs})
    
    end_time = current_time()
    print("EMBENDING SYMPTOMS End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("EMBENDING SYMPTOMS Execution time: " + str(time_diff))
    
    return embending_symptoms_dict


def embending_diagnosis(model, admissions, config=LEGACY):
    """Build the baseline arm's diagnosis embedding dict.

    Note the dict is keyed by the RAW description while the embedded string is
    the preprocessed one -- that asymmetry predates this change and is what
    lets get_diagnosis_similarity_by_description_max look up by raw text. Only
    the embedded string is affected by ``config``, so FIX 3a reaches DRG names
    like 'Tracheostomy w/o Extensive Procedure' without disturbing the keys.
    """
    start_time = current_time()
    print("EMBENDING DIAGNOSIS Start time: ", start_time)
    embending_diagnosis_dict = dict()

    for key in admissions:
        a = admissions.get(key)
        diagnosis_list = a.diagnosis
        for d in diagnosis_list:
            diagnosis_description = d[d.index(':') + 1:len(d)]
            embs = model.embed_sentence(preprocess_sentence(diagnosis_description, config))
            embending_diagnosis_dict.update({diagnosis_description: embs})
    
    end_time = current_time()
    print("EMBENDING DIAGNOSIS End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("EMBENDING DIAGNOSIS Execution time: " + str(time_diff))
    
    return embending_diagnosis_dict


def load_model():
    # sent2vec is imported here, not at module scope, so the BERT arm can import
    # this module without it. pyproject.toml keeps sent2vec in the `baseline`
    # extra precisely so a BERT-only install never needs it; a module-scope
    # import would silently break that contract. See docs/guides/setup.md.
    import sent2vec

    start_time = current_time()
    print("LOAD MODEL Start time: ", start_time)
    model = sent2vec.Sent2vecModel()
    model.load_model(os.getcwd() + '/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
    
    end_time = current_time()
    print("LOAD MODEL End time: ", end_time)
    time_diff = elapsed_time(start_time, end_time)
    print("LOAD MODEL Execution time: " + str(time_diff))
    
    return model


def preprocess_sentence(text, config=LEGACY):
    """Lowercase, pad punctuation, tokenise, drop punctuation and stopwords.

    ``config`` defaults to LEGACY, so every existing call site -- including the
    ones inside this module and the golden's run_analysis path -- produces the
    exact same string it did before this parameter existed.

    Under CORRECTED the only difference is FIX 3a: 'w/o' is expanded to
    'without' *before* the slash-padding runs, so the negation reaches the
    encoder instead of being tokenised away. See _W_SLASH_O above for why the
    rewrite happens here rather than in the stopword filter, and why the word
    boundaries matter. Everything downstream of that line is byte-identical
    between the two versions.
    """
    if use_corrected_preprocessing(config):
        text = _W_SLASH_O.sub('without', text)

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


def get_diagnosis_similarity_by_description_max_model(model, gt_diagnosis, predicted_diagnosis, method, config=LEGACY):
    MIN_SIMILARITY = 0
    max_diagnosis_similarity = dict()
    max_similarity = MIN_SIMILARITY
    
    for x in gt_diagnosis:
        x_description = x[x.index(':') + 1:len(x)]
        for y in predicted_diagnosis:
            y_description = y[y.index(':') + 1:len(y)]
            emb_diagnosis_to_predict = model.embed_sentence(preprocess_sentence(x_description, config))
            emb_diagnosis_predicted = model.embed_sentence(preprocess_sentence(y_description, config))
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
