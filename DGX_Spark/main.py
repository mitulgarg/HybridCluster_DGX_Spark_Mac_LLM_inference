"""
DGX/Server: Full precision verifier using llama.cpp
Install: pip install llama-cpp-python fastapi uvicorn
Download model: wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
Run: uvicorn main:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from llama_cpp import Llama

app = FastAPI()

# Load full precision model (Q8_0 = near FP16 quality)
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=99, verbose=False, logits_all=True)

# State
current_tokens = []

class PrefillRequest(BaseModel):
    prompt: str

class VerifyRequest(BaseModel):
    draft_tokens: List[int]

@app.post("/prefill/")
def prefill(req: PrefillRequest):
    global current_tokens
    prompt_tokens = llm.tokenize(req.prompt.encode('utf-8'))
    llm.reset()
    llm.eval(prompt_tokens)
    current_tokens = prompt_tokens
    print(f"Prefill: {len(prompt_tokens)} tokens")
    return {"prompt_len": len(prompt_tokens), "prompt_ids": prompt_tokens}

@app.post("/verify/")
def verify(req: VerifyRequest):
    global current_tokens
    draft_tokens = req.draft_tokens
    if not draft_tokens:
        return {"accepted_prefix_len": 0, "preds": []}
    
    print(f"\nReceived draft tokens: {draft_tokens}")
    print(f"Current context length: {len(current_tokens)}")
    
    # Generate what the server would predict
    prompt_text = llm.detokenize(current_tokens).decode('utf-8', errors='ignore')
    
    # Generate the same number of tokens the draft sent
    output = llm(
        prompt_text,
        max_tokens=len(draft_tokens),
        temperature=0.0,  # Greedy
        echo=False
    )
    
    generated_text = output['choices'][0]['text']
    server_tokens = llm.tokenize(generated_text.encode('utf-8'))
    
    # Filter special tokens
    server_tokens = [t for t in server_tokens if t not in [0, 1, 2]]
    
    print(f"Server would generate: {server_tokens[:5]}... (draft: {draft_tokens[:5]}...)")
    
    # Compare token by token
    accepted = 0
    for i, draft_token in enumerate(draft_tokens):
        if i < len(server_tokens) and server_tokens[i] == draft_token:
            accepted += 1
        else:
            break
    
    # Update state with accepted tokens
    if accepted > 0:
        current_tokens.extend(draft_tokens[:accepted])
        accepted_text = llm.detokenize(draft_tokens[:accepted]).decode('utf-8', errors='ignore')
        print(f"Accepted: {accepted}/{len(draft_tokens)} tokens: {repr(accepted_text)[:50]}")
    else:
        print(f"Rejected all. Server first token: {server_tokens[0] if server_tokens else 'None'}")
    
    return {
        "accepted_prefix_len": accepted, 
        "preds": server_tokens[:len(draft_tokens)],  # Return what server predicted
        "total_tokens": len(current_tokens)
    }

@app.get("/")
def root():
    return {"status": "running", "model": MODEL_PATH, "tokens": len(current_tokens)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)