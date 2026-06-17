import requests
import json

base_url = 'http://localhost:8000'

print('1. Testing Root Endpoint:')
r1 = requests.get(base_url + '/')
print(r1.status_code, r1.json() if r1.status_code == 200 else r1.text)

print('\n2. Testing History Endpoint:')
r2 = requests.get(base_url + '/api/v1/history')
print(r2.status_code, len(r2.json()) if r2.status_code == 200 else r2.text)

print('\n3. Testing Predict Endpoint (Real World):')
payload = {
    'meter_id': 'TEST_001',
    'readings': [1.5, 1.6, 1.4, 0.0, 0.0, 1.2, 1.3] * 5,
    'model_type': 'real_world'
}
r3 = requests.post(base_url + '/api/v1/predict', json=payload)
print(r3.status_code, r3.json() if r3.status_code == 200 else r3.text)

print('\n4. Testing Explain Endpoint (Real World):')
r4 = requests.post(base_url + '/api/v1/explain', json=payload)
print(r4.status_code, 'Success' if r4.status_code == 200 else r4.text)

