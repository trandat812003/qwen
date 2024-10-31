import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.sample_rate = 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        MODEL_PRETRAIN = "Qwen/Qwen-Audio"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PRETRAIN, device_map="cpu", trust_remote_code=True, bf16=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)

        model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")
        self.output_type = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()
            input_strings = [text.decode("utf-8") for text in input_data]

            print("Input data:", input_strings)

            audio_url = f'./data/{input_strings[0]}'
            print(audio_url)
            # audio_url = "./data/dia1_utt0.wav"
            sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
            query = f"<audio>{audio_url}</audio>{sp_prompt}"

            audio_info = self.tokenizer.process_audio(query)
            inputs = self.tokenizer(query, return_tensors='pt', audio_info=audio_info)
            inputs = inputs.to(self.model.device)
            pred = self.model.generate(**inputs, audio_info=audio_info)
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False,audio_info=audio_info)

            output_data = np.array([response], dtype=object)
            output_tensor = pb_utils.Tensor("OUTPUT", output_data.astype(self.output_type))

            inference_response = pb_utils.InferenceResponse([output_tensor])
            responses.append(inference_response)

        return responses
    
    def finalize(self):
        print('Cleaning up...')
