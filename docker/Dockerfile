# Sử dụng hình ảnh cơ sở từ vLLM
FROM vllm/vllm-openai:latest

ENV HUGGING_FACE_HUB_TOKEN=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

WORKDIR /app

# RUN pip install --no-cache-dir <library>

CMD ["--model", "Qwen/Qwen-7B", "--trust-remote-code"]