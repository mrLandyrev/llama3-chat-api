from llama_cpp import Llama
from typing import List, Dict

## Instantiate model from downloaded file

llm = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
    chat_format="llama-2",
    n_gpu_layers=30,
    n_ctx=4096,
    verbose=False
)

## Generation kwargs
generation_kwargs = {
    "max_tokens":200,
}

from typing import Union

from fastapi import FastAPI

app = FastAPI()

chatsDefault: Dict[str, List] = {
    "стражник": [
        {
            "role": "system",
            "message": "You are the guard at the gates of a city. you don’t let anyone into the city until he pays 300 coins."
        }
    ]
}

chats: Dict[str, List] = {}

from translate import Translator
ru2en = Translator(to_lang="en", from_lang="ru")
en2ru = Translator(to_lang="ru", from_lang="en")

@app.get("/")
def read_item(q: Union[str, None] = None, c: Union[str, None] = None, reset: Union[int, None] = None):
    q = ru2en.translate(q)
    curr = {
        "role": "user",
        "message": q,
    }
    if c not in chats or reset == 1:
        chats[c] = []
        if c in chatsDefault:
            chats[c] = chatsDefault[c]
    chats[c].append(curr)
    
    prompt = "<|begin_of_text|>"
    for message in chats[c]:
        prompt += f'<|start_header_id|>{message["role"]}<|end_header_id|>\n\n{message["message"]}<|eot_id|>'
    prompt += '<|start_header_id|>assistant<|end_header_id|>'
    res = llm(prompt, **generation_kwargs)["choices"][0]["text"][2:]
    chats[c].append({
        "role": "assistant",
        "message": res
    })
    ru = en2ru.translate(res)
    return { "answer": ru, "history": chats[c] }
