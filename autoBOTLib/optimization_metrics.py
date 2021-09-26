import numpy as np
from sklearn.metrics import confusion_matrix


def get_metric_report(y_true, y_prediction):
    """
    A generic metric report; suitable for multiobjective experiments (not the core paper)
    """

    cnf_matrix = confusion_matrix(y_true, y_prediction)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TP / (TP + FN)

    # Specificity or true negative rate
    TN / (TN + FP)

    # Precision or positive predictive value
    TP / (TP + FP)

    # Negative predictive value
    TN / (TN + FN)

    # Fall out or false positive rate
    FP / (FP + TN)

    # False negative rate
    FN / (TP + FN)

    # False discovery rate
    FP / (TP + FP)

    return np.mean(TP), np.mean(TN), np.mean(FP), np.mean(FN)
