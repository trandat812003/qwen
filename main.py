from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch


torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", device_map="cpu", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)


audio_url = "./data/dia1_utt0.wav"
sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
query = f"<audio>{audio_url}</audio>{sp_prompt}"


audio_info = tokenizer.process_audio(query)
inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info)
inputs = inputs.to(model.device)
pred = model.generate(**inputs, audio_info=audio_info)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False,audio_info=audio_info)
print(response)