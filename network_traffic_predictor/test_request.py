import requests

test_data = {
    "sequence": [10.2, 9.8, 10.5, 11.2, 12.1, 12.8, 12.5, 11.9, 11.2, 10.5, 9.8, 9.2, 8.9, 9.1, 9.8, 10.5, 11.2, 11.9, 12.5, 12.8, 12.1, 11.2, 10.5, 9.8],
    "hours_ahead": 12
}

response = requests.post('http://localhost:5000/predict', json=test_data)
print("Status code:", response.status_code)
print("Response:", response.text)