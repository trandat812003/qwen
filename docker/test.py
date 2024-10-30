import tritonclient.http as httpclient
import numpy as np

def test_triton_model():
    url = "localhost:8000"
    model_name = "qwen"


    client = httpclient.InferenceServerClient(url=url)

    input_text = "test 123 tesst -- 11 -- ++."
    input_data = np.array([input_text], dtype="object")

    input_tensor = httpclient.InferInput("INPUT", input_data.shape, "BYTES")
    input_tensor.set_data_from_numpy(input_data)

    response = client.infer(
        model_name=model_name,
        inputs=[input_tensor]
    )

    output_data = response.as_numpy("OUTPUT")
    print("Output từ mô hình Triton:", output_data[0].decode("utf-8"))

test_triton_model()
