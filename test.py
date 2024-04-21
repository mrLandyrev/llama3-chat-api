from llama_cpp import Llama

## Instantiate model from downloaded file

llm = Llama(
    model_path="Meta-Llama-3-8B-Instruct.Q8_0.gguf",
    n_gpu_layers=30,
    n_ctx=4096,
    verbose=False,
)

## Generation kwargs
generation_kwargs = {
    "max_tokens":200,
}

messages = [
    {
        "role": "system",
        "message": "You are helpfull assistant"
    },
    {
        "role": "user",
        "message": "Hi, i am Sergey!"
    },
]

prompt = "<|begin_of_text|>"
for message in messages:
    prompt += f'<|start_header_id|>{message["role"]}<|end_header_id|>\n\n{message["message"]}<|eot_id|>'
prompt += '<|start_header_id|>assistant<|end_header_id|>'

print(llm(prompt, **generation_kwargs)["choices"][0]['text'])
