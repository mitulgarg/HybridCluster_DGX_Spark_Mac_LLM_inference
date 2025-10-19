# main.py
# Run with: uvicorn main:app --host 127.0.0.1 --port 8000

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from prefill import router as prefill_router
from verifier import router as verifier_router

app = FastAPI(title="DGX Verifier (simulated)")

# === Model and tokenizer shared globally ===
MODEL_NAME = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"    #this is because the POC is done using CPU with  Mac mini M4

print(f"Loading verifier model {MODEL_NAME} on {device} ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Shared global state (you can later move this to a class or Redis)
app.state.model = model
app.state.tokenizer = tokenizer
app.state.device = device
app.state.kv_cache = None
app.state.prompt_input_ids = None
app.state.prompt_len = 0

# === Routers ===
app.include_router(prefill_router, prefix="/prefill", tags=["Prefill"])
app.include_router(verifier_router, prefix="/verify", tags=["Verify"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
