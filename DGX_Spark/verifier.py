# verifier.py

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List
import torch

router = APIRouter()

class VerifyRequest(BaseModel):
    draft_tokens: List[int]

@router.post("/")
def verify(req: VerifyRequest, request: Request):
    """
    Verify draft token batch using stored KV cache.
    Returns accepted_prefix_len: number of leading tokens that match
    the full-model top-1 predictions.
    """
    model = request.app.state.model
    device = request.app.state.device
    kv_cache = request.app.state.kv_cache
    prompt_len = request.app.state.prompt_len

    if kv_cache is None:
        return {"error": "No KV cache available. Call /prefill first."}

    draft_tokens = req.draft_tokens
    if len(draft_tokens) == 0:
        return {"accepted_prefix_len": 0}

    draft_tensor = torch.tensor([draft_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(draft_tensor, past_key_values=kv_cache, use_cache=True)
        logits = outputs.logits

    preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

    accepted = 0
    for i, (pred_token, draft_token) in enumerate(zip(preds, draft_tokens)):
        if pred_token == draft_token:
            accepted += 1
        else:
            break

    if accepted > 0:
        request.app.state.prompt_len = prompt_len + accepted
        # Optionally recompute/append to KV cache (omitted for simplicity)

    return {"accepted_prefix_len": accepted, "preds": preds}
