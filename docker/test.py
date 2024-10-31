import requests
import numpy as np
import json

def test_triton_model():
    url = "http://localhost:8000/v2/models/qwen/infer"
    input_text = "test 123 tesst -- 11 -- ++."

    input_data = np.array([input_text], dtype="object").tolist()

    payload = {
        "inputs": [
            {
                "name": "INPUT",
                "shape": [1],
                "datatype": "BYTES",
                "data": input_data
            }
        ],
        "outputs": [{"name": "OUTPUT"}]
    }

    response = requests.post(url, json=payload, timeout=3600)

    if response.status_code == 200:
        output_data = response.json()["outputs"][0]["data"][0]
        print("Output từ mô hình Triton:", output_data)
    else:
        print("Lỗi:", response.status_code, response.text)

test_triton_model()
