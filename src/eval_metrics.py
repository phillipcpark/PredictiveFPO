from params import *

from collections import Counter
import numpy as np
import sys

# evaluate predictions
def pred_acc(predicts, labels):

    correct = total = 0
    for p_idx in range(len(predicts)):
        if (labels[p_idx].detach() == IGNORE_CLASS):
            continue  

        pred_class = None

        #FIXME thresholded prediction
        if (USE_PRED_THRESH):
            pred_class = 0
            max_prob   = np.amax(predicts.detach().numpy()[p_idx])
            if (max_prob >= PRED_THRESH):
                pred_class = np.argmax(predicts[p_idx].detach().numpy())

        else:
            pred_class = np.argmax(np.array(predicts[p_idx].detach())) 
        true_class = np.array(labels[p_idx].detach())

        if (pred_class == true_class):
            correct += 1
        total += 1            

    return float(correct) / total

#
def prec_recall(predicts, labels):
   
    correct   = {}
    incorrect = {}
    true      = Counter(labels.detach().numpy())
    for c in range(CLASSES):
        correct[c] = 0
        incorrect[c] = 0

    for p_idx in range(len(predicts)):
        gt   = labels[p_idx].detach().numpy() 

        pred_class = 0

        if (USE_PRED_THRESH):
            max_prob   = np.amax(predicts.detach().numpy()[p_idx])
            if (max_prob >= PRED_THRESH):
                pred_class = np.argmax(predicts[p_idx].detach().numpy())       
        else:
            pred_class = np.argmax(predicts[p_idx].detach().numpy()) 

        if (pred_class == gt):
            correct[pred_class] += 1
        else:
            incorrect[pred_class] += 1             

    total = len(labels)

    prec = {}
    rec  = {}

    for c in range(CLASSES):

        if (true[c] == 0):
            if (incorrect[c] == 0):
                prec[c] = 1.0
                rec[c]  = 1.0
            else:
                prec[c] = 0.0
                rec[c]  = 1.0
        else:
            if (correct[c] + incorrect[c] == 0):
                prec[c] = 0.0
                rec[c]  = 0.0
            else:
                prec[c] = float(correct[c]) / (correct[c] + incorrect[c])
                rec[c] = float(correct[c]) / true[c] 
    return prec, rec



# 
def relative_error(val_num, val_denom):       
    try:
        return abs((val_num - val_denom) / val_denom)
    except:
        return 0.0


# see if error sample is acceptable
def accept_err(errs):
    sz           = len(errs)
    contrib_prop = 1.0 / sz
    prop_over_thresh = 0.0

    for err in errs:
        if (err > err_thresh):
            prop_over_thresh += contrib_prop
            if (prop_over_thresh > (1.0 - err_accept_prop)):
                return False, prop_over_thresh
    return True, prop_over_thresh

