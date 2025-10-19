# main.py
# Run with: uvicorn main:app --host 127.0.0.1 --port 8000

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="DGX Verifier (simulated)")

# === Model and tokenizer shared globally ===
MODEL_NAME = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading verifier model {MODEL_NAME} on {device} ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Shared global state
app.state.model = model
app.state.tokenizer = tokenizer
app.state.device = device
app.state.kv_cache = None
app.state.current_tokens = []  # Track all generated tokens

# === Prefill Endpoint ===
from pydantic import BaseModel
from typing import List

class PrefillRequest(BaseModel):
    prompt: str

@app.post("/prefill/")  # Add trailing slash to avoid redirect
def prefill(req: PrefillRequest):
    """Run prefill and store KV cache"""
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    # Store KV cache and tokens
    app.state.kv_cache = outputs.past_key_values
    app.state.current_tokens = input_ids.squeeze(0).cpu().tolist()
    
    return {
        "status": "ok",
        "prompt_len": len(app.state.current_tokens),
        "prompt_ids": app.state.current_tokens,
    }

# === Verify Endpoint ===
class VerifyRequest(BaseModel):
    draft_tokens: List[int]

@app.post("/verify/")  # Add trailing slash to avoid redirect
def verify(req: VerifyRequest):
    """Verify draft tokens and update KV cache for accepted tokens"""
    if app.state.kv_cache is None:
        return {"error": "No KV cache available. Call /prefill first."}
    
    draft_tokens = req.draft_tokens
    if len(draft_tokens) == 0:
        return {"accepted_prefix_len": 0, "preds": []}
    
    # Convert draft tokens to tensor
    draft_tensor = torch.tensor([draft_tokens], dtype=torch.long).to(device)
    
    # Run verification with cached KV
    with torch.no_grad():
        outputs = model(draft_tensor, past_key_values=app.state.kv_cache, use_cache=True)
        logits = outputs.logits
    
    # Get predictions (argmax for greedy)
    preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
    
    # Count accepted tokens (consecutive matches)
    accepted = 0
    for pred_token, draft_token in zip(preds, draft_tokens):
        if pred_token == draft_token:
            accepted += 1
        else:
            break
    
    # CRITICAL FIX: Update KV cache with accepted tokens
    if accepted > 0:
        accepted_tensor = torch.tensor([draft_tokens[:accepted]], dtype=torch.long).to(device)
        with torch.no_grad():
            new_outputs = model(accepted_tensor, past_key_values=app.state.kv_cache, use_cache=True)
            app.state.kv_cache = new_outputs.past_key_values  # Update cache
            app.state.current_tokens.extend(draft_tokens[:accepted])
    
    return {
        "accepted_prefix_len": accepted,
        "preds": preds,
        "total_tokens": len(app.state.current_tokens)
    }

# === Health Check ===
@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": str(device),
        "tokens_generated": len(app.state.current_tokens) if app.state.current_tokens else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)