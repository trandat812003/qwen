from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-Audio",
    device_map="cpu", 
    trust_remote_code=True,
    torch_dtype=torch.float32 
).eval()

audio_url = "./dia1_utt0.wav"
sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
query = f"<audio>{audio_url}</audio>{sp_prompt}"
audio_info = tokenizer.process_audio(query)
inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info)
inputs = inputs.to(model.device)

try:
    pred = model.generate(**inputs, audio_info=audio_info)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False, audio_info=audio_info)
    print(response)
except NotImplementedError as e:
    print("Error:", e)
