from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
llm = LLM(
    model="Qwen/Qwen-7B", 
    tensor_parallel_size=1, 
    trust_remote_code=True, 
    dtype=torch.float16, 
    device='cpu'
)


audio_url = "./dia1_utt0.wav"
sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
query = f"<audio>{audio_url}</audio>{sp_prompt}"


outputs = llm.generate([query], sampling_params)
for output in outputs:
    print("Generated text:", output.text)
