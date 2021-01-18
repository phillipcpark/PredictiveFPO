from params import *
import numpy as np

# evaluate predictions
def pred_acc(predicts, labels):

    correct = total = 0
    for p_idx in range(len(predicts)):
        if (labels[p_idx].detach() == IGNORE_CLASS):
            continue  

        pred_class = np.argmax(np.array(predicts[p_idx].detach())) 
        true_class = np.array(labels[p_idx].detach())

        if (pred_class == true_class):
            correct += 1
        total += 1            

    return float(correct) / total


# 
def relative_error(val_num, val_denom):       
    try:
        return abs((val_num - val_denom) / val_denom)
    except:
        return 0.0

