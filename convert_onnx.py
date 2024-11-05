import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

MODEL_PRETRAIN = "Qwen/Qwen-Audio"


bnb_config = BitsAndBytesConfig(load_in_8bit=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAIN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PRETRAIN,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="cuda:6"
)


model.eval()


input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda:6')


onnx_file_path = "qwen_audio_model.onnx"
torch.onnx.export(
    model,
    input_ids,
    onnx_file_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Mô hình đã được xuất sang định dạng ONNX và lưu tại: {onnx_file_path}")
