from transformers import export, AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen-Audio"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Đường dẫn lưu file ONNX
onnx_path = "qwen_audio_model.onnx"
export(model, onnx_path, tokenizer=tokenizer)