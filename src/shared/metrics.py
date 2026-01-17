"""
Performance metrics computation - confusion matrix, precision, recall, F-score.
"""

from src.shared.constants import PERFORMANCE_INDEX_HEADER, TP, FP, P, R, FS, PR, K_FOLD


def init_confusion_matrix():
    """
    Initialize confusion matrix for different similarity thresholds.
    Structure: { threshold : [TP, FP] }
    """
    confusion_matrix = dict()
    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        confusion_matrix.update({i: [0 for x in range(2)]})
    return confusion_matrix


def init_performance_matrix():
    """
    Initialize performance matrix for different similarity thresholds.
    Structure: { threshold : [TP, FP, P, R, FS, PR] }
    """
    confusion_matrix = dict()
    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        confusion_matrix.update({i: [0 for x in range(6)]})
    return confusion_matrix


def containGreaterOrEqualsValue(topK, top_similarities, b):
    """
    Check if any of the top-K similarities is >= threshold b.
    """
    for i in range(0, topK):
        if i < len(top_similarities) and top_similarities[i] >= b:
            return True
    return False


def compute_performance_index(confusion_matrix, nrow, performance_out_file):
    """
    Compute and write performance metrics from confusion matrix.
    """
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
    """
    Compute and aggregate performance metrics across folds.
    """
    performance_out_file.write(PERFORMANCE_INDEX_HEADER)

    for i in {1, 0.9, 0.8, 0.7, 0.6}:
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
    """
    Print final averaged performance metrics.
    """
    performance_out_file.write(PERFORMANCE_INDEX_HEADER)

    for i in {1, 0.9, 0.8, 0.7, 0.6}:
        values = performance_matrix.get(i)
        performance_out_file.write(
            str(i) + "\t" + str(values[TP] / K_FOLD) + "\t" + str(values[FP] / K_FOLD) + "\t" + str(
                values[P] / K_FOLD) + "\t" + str(values[R] / K_FOLD) +
            "\t" + str(values[FS] / K_FOLD) + "\t" + str(values[PR] / K_FOLD) + "\n")
