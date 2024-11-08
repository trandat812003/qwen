import os
import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

wer_metric = load("wer", trust_remote_code=True)
bleu_metric = load("bleu", trust_remote_code=True)
rouge_metric = load("rouge", trust_remote_code=True)

data_folder = './data/'


torch.cuda.empty_cache()

ground_truth = {}
with open(os.path.join(data_folder, 'result.txt'), 'r') as f:
    for line in f:
        file_id, text = line.strip().split(' ', 1)
        ground_truth[file_id] = text

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
MODEL_PRETRAIN = "Qwen/Qwen-Audio"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PRETRAIN,
    trust_remote_code=True,
    device_map="cuda:0",
    quantization_config=bnb_config,
)
model.generation_config = GenerationConfig.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
print(model.device)

hypotheses = []
references = []
with torch.no_grad():
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


wer = wer_metric.compute(predictions=hypotheses, references=references)
bleu = bleu_metric.compute(predictions=hypotheses, references=references)
rouge = rouge_metric.compute(predictions=hypotheses, references=references)


print("WER:", wer)
print("BLEU:", bleu)
print("ROUGE:", rouge)
