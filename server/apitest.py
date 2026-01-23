import requests

url = "http://localhost:8090/query"

data = {
    "query": ["What is the capital of France?"],
    "retrieval_num": 5
}

response = requests.post(url, json=data)
print(response.json()["result"][0])