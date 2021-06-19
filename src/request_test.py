import requests
import json

PORT = 5001

URL     = 'http://0.0.0.0:' + str(PORT) + '/predict'
headers = {'content-type': 'application/json'}

post_data = {'node_features':[5.9, 3.0, 5.1, 1.8], 'edges': [1,2,3]}
response  = requests.post(URL, json=post_data, headers=headers)

try:
    r_payload = json.loads(response.text)
    print(r_payload['node_features'])
except:
    print(response)
