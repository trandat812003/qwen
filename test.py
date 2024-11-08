import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

data_folder = './abc/'


torch.cuda.empty_cache()

MODEL_PRETRAIN = "Qwen/Qwen-Audio"

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PRETRAIN,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="cuda:0"
)
model.generation_config = GenerationConfig.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
print(model.device)

prompts = {
    '1': "<|startofanalysis|><|unknown|><|audio_grounding|><|en|><|notimestamps|>",
    '2': "<|startofanalysis|><|unknown|><|emotion_recognition|><|en|><|notimestamps|>",
    '3': "<|startofanalysis|><|unknown|><|event|><|en|><|notimestamps|>",
    '4': "<|startofanalysis|><|unknown|><|instrument|><|en|><|notimestamps|>",
    '5': "<|startofanalysis|><|unknown|><|music_description|><|en|><|notimestamps|>",
    '6': "<|startofanalysis|><|unknown|><|question|>Can people be heard talking?<|en|><|notimestamps|>",
    '7': "<|startofanalysis|><|unknown|><|scene|><|en|><|notimestamps|>",
    '8': "<|startofanalysis|><|unknown|><|sonic|><|en|><|notimestamps|>",
    '9': "<|startoftranscript|><|en|><|speaker_meta|><|en|><|notimestamps|>",
    '10': "<|startoftranscript|><|en|><|speech_understanding|><|en|><|notimestamps|>"
}


with torch.no_grad():
    for file in os.listdir(data_folder):
        file_name = os.path.splitext(file)[0]

        audio_file = os.path.join(data_folder, file)

        query = f"<audio>{audio_file}</audio>{prompts.get(file_name)}"
        audio_info = tokenizer.process_audio(query)
        inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info).to(model.device)

        torch.cuda.empty_cache()

        pred = model.generate(**inputs, audio_info=audio_info)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True, audio_info=audio_info)

        print(response)

        torch.cuda.empty_cache()
