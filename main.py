import os
import torch
from datasets import load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

metric = load_metric("wer")

data_folder = './data/121123/'


torch.cuda.empty_cache()

ground_truth = {}
with open(os.path.join(data_folder, '84-121123.trans.txt'), 'r') as f:
    for line in f:
        file_id, text = line.strip().split(' ', 1)
        ground_truth[file_id] = text

MODEL_PRETRAIN = "Qwen/Qwen-Audio"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PRETRAIN,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="cuda:6"
)
model.generation_config = GenerationConfig.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
print(model.device)

hypotheses = []
references = []
for file_id, text in ground_truth.items():
    audio_file = os.path.join(data_folder, f'{file_id}.flac')

    sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
    query = f"<audio>{audio_file}</audio>{sp_prompt}"
    audio_info = tokenizer.process_audio(query)
    inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info).to(model.device)

    torch.cuda.empty_cache()

    pred = model.generate(**inputs, audio_info=audio_info)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True, audio_info=audio_info)

    hypotheses.append(response)
    references.append(text)

    print(f"File: {file_id}, Ground Truth: {text}")
    print(f"Prediction: {response}\n")

wer = metric.compute(predictions=hypotheses, references=references)
print(wer)
