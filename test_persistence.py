
import requests
import time
import base64

url = "http://localhost:8803/synthesize"
payload = {
    "sentence": "This is a test of persistent memory residency.",
    "voice": "af_heart",
    "speed": 1.0
}

for i in range(3):
    start = time.time()
    print(f"Request {i+1} sending...")
    resp = requests.post(url, json=payload)
    end = time.time()
    print(f"Request {i+1}: StatusCode={resp.status_code}, Time={end-start:.4f}s")
    if resp.status_code == 200:
        data = resp.json()
        if "base64" in data:
            print(f"  Got base64 length: {len(data['base64'])}")
        else:
            print("  No base64 field")
    else:
        print("  Error:", resp.text)
