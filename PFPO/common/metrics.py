from mpmath import mp
from collections import Counter
from numpy import amax, argmax

#
def prec_recall(predicts, labels, classes, ignore_class, pred_thresh=None):
    correct   = {}
    incorrect = {}
    true      = {0:0, 1:0} 

    for c in range(classes):
        correct[c] = 0
        incorrect[c] = 0

    for p_idx in range(len(predicts)):
        gt   = int(labels[p_idx].detach().numpy()) 

        if (gt == ignore_class):
            continue 
        true[gt]  += 1
        pred_class = None

        if not(pred_thresh is None):
            max_prob   = amax(predicts.detach().numpy()[p_idx])
            if (max_prob >= pred_thresh):
                pred_class = argmax(predicts[p_idx].detach().numpy())       
            else:
                pred_class = 0
        else:
            pred_class = argmax(predicts[p_idx].detach().numpy()) 

        if (pred_class == gt):
            correct[pred_class] += 1
        else:
            incorrect[pred_class] += 1             

    prec = {}
    rec  = {}
    for c in range(classes):
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
    mp.prec = 65     
    try:
        return mpmath.fabs((val_num - val_denom) / val_denom)
    except:
        return 0.0


# see if error sample is acceptable
def accept_err(errs, thresh, accept_proportion):
    sz        = len(errs)
    gt_thresh = 0
    max_count = int(sz * (1.0 - accept_proportion))

    accept_thresh = thresh

    for err in errs:       
        if (err > accept_thresh):
            gt_thresh += 1

    prop_gt_thresh = float(gt_thresh) / float(sz) 
    if (prop_gt_thresh >= (1.0 - accept_proportion)):
        return False, prop_gt_thresh
    return True, prop_gt_thresh

#
def update_freq_delta(prec_freq_deltas, otc_tuned, otc_orig):    
    # precision counts
    total = len(otc_tuned)

    orig_counts = {32:0, 64:0, 80:0}
    for prec in otc_orig:
        orig_counts[prec] += 1

    prec_counts = {32:0, 64:0, 80:0} 
    for prec in otc_tuned:
        prec_counts[prec] += 1                

    for k in prec_counts.keys():
        prec_freq_deltas[k].append(float(prec_counts[k])/total - float(orig_counts[k])/total)




