# prefill.py

from fastapi import APIRouter, Request
from pydantic import BaseModel
import torch

router = APIRouter()

class PrefillRequest(BaseModel):
    prompt: str

@router.post("/")
def prefill(req: PrefillRequest, request: Request):
    """
    Run a forward pass on the prompt and store past_key_values (KV cache).
    Returns the prompt token length so drafter can set its cursor.
    """
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device

    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    # Store KV cache and prompt info in app state
    request.app.state.kv_cache = outputs.past_key_values
    request.app.state.prompt_input_ids = input_ids.squeeze(0).cpu().tolist()
    request.app.state.prompt_len = input_ids.shape[1]

    return {
        "status": "ok",
        "prompt_len": request.app.state.prompt_len,
        "prompt_ids": request.app.state.prompt_input_ids,
    }
