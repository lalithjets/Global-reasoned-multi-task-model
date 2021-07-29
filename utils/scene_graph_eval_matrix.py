import numpy as np
import sklearn.metrics

def compute_mean_avg_prec(y_true, y_score):
    try:
        avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
        mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
    except ValueError:
        mean_avg_prec = 0

    return mean_avg_prec

def calibration_metrics(logits_all, labels_all):
    
    logits = logits_all.detach().cpu().numpy()
    labels = labels_all.detach().cpu().numpy()
    map_value = compute_mean_avg_prec(labels, logits)
    labels = np.argmax(labels, axis=-1)
    recall = sklearn.metrics.recall_score(labels, np.argmax(logits,1), average='macro')
    return(map_value, recall)