import requests
import numpy as np
import subprocess
import json


def copy_file_to_container(file_path, container_id='test'):
    try:
        subprocess.run(["docker", "cp", f'./data/{file_path}', f"{container_id}:/opt/tritonserver/data/{file_path}"], check=True)
        print(f"done")
    except subprocess.CalledProcessError as e:
        print(e)


def test_triton_model():
    url = "http://localhost:8000/v2/models/qwen/infer"
    input_text = "dia1_utt0.wav"

    copy_file_to_container(input_text)

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
