import requests

base_url = "http://localhost:5005"
endpoints = [
    "/",
    "/predict",
    "/health",
    "/docs", 
    "/swagger",
    "/invocations",
    "/ping",
    "/v1/models",
    "/model",
    "/api/predict"
]

print("Testing available endpoints:")
for endpoint in endpoints:
    try:
        response = requests.get(f"{base_url}{endpoint}", timeout=5)
        print(f"{endpoint}: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.text[:200]}...")
    except Exception as e:
        print(f"{endpoint}: Error - {e}")
    print("---")