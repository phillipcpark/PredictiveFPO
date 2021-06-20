from params import *
from train_test_bignn import get_dev, batch_graphs_from_idxs
from bignn import bignn
from dgl import batch

from json import dumps
import torch as th
from flask import Flask, request

app = Flask(__name__)

#
def load_bignn(path=None):
    global m

    if (MOD_PATH is None):
        m = bignn(feat_dim, H_DIM, CLASSES)
        return m

    mod_dev    = get_dev() if USE_GPU else th.device('cpu')
    state_dict = th.load(MOD_PATH, map_location=mod_dev)

    m = bignn(OP_ENC_DIM, H_DIM, CLASSES)
    m.to(mod_dev)
    m.load_hier_state(state_dict)
    return m 


#
@app.route('/predict', methods=['POST'])
def bignn_predict():         
    prog_g      = request.get_json()      
    n_feats     = prog_g['node_features']
    g_edges     = prog_g['graph_edges']
    unary_masks = prog_g['unary_masks']
    prog_g      = batch_graphs_from_idxs([0], [g_edges], [unary_masks], [0], [n_feats], USE_GPU) 
 
    predicts, _ = m(prog_g, USE_GPU)
    sm          = th.nn.Softmax(dim=-1)
    predicts    = sm(th.sigmoid(predicts))
    predicts    = predicts.tolist()

    recs = [1 if p[1] > PRED_THRESH else 0 for p in predicts] 
    return dumps({'tune_recommendations': recs})

#
#
#
if __name__ == '__main__':
    model = load_bignn(MOD_PATH)
    app.run(host='0.0.0.0', port=5001)

