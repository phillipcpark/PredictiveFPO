import requests
import json
import sys

from predictiveFPO import ld_predfpo_ds

PORT    = 5001
URL     = 'http://0.0.0.0:' + str(PORT) + '/predict'
headers = {'content-type': 'application/json'}

#
#
#
if __name__=='__main__':
    if not(len(sys.argv) == 2):
        raise RuntimeError('Usage: <ds directory path>')
    path = sys.argv[1]
    ds   = ld_predfpo_ds(path)

    for i in range(2):      
        post_data = {'node_features':ds['feats'][i], 'graph_edges': ds['g_edges'][i], 'unary_masks':ds['unary_masks'][i]}
        response  = requests.post(URL, json=post_data, headers=headers)
        
        try:
            r_payload = json.loads(response.text)
            print(r_payload['tune_recommendations'])
        except:
            print('Unexpected response: ' + str(response))





