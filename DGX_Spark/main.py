# main.py - CORRECTED Speculative Decoding Verifier
# Run with: uvicorn main:app --host 127.0.0.1 --port 8000

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel
from typing import List
import copy

app = FastAPI(title="DGX Verifier")

MODEL_NAME = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading verifier model {MODEL_NAME} on {device} ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

app.state.model = model
app.state.tokenizer = tokenizer
app.state.device = device
app.state.kv_cache = None
app.state.current_tokens = []

DEBUG = True

class PrefillRequest(BaseModel):
    prompt: str

@app.post("/prefill/")
def prefill(req: PrefillRequest):
    """Run prefill and cache KV"""
    if DEBUG:
        print(f"\n=== PREFILL ===")
        print(f"Prompt: {req.prompt}")
    
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    app.state.kv_cache = outputs.past_key_values
    app.state.current_tokens = input_ids.squeeze(0).cpu().tolist()
    app.state.prefill_logits = outputs.logits  # Save for first prediction
    
    if DEBUG:
        print(f"Tokens: {app.state.current_tokens}")
        print(f"KV shape: {outputs.past_key_values[0][0].shape}")
        # What would WE generate next?
        next_pred = torch.argmax(outputs.logits[0, -1, :]).item()
        print(f"Our next token: {next_pred} ({repr(tokenizer.decode([next_pred]))})")
    
    return {
        "status": "ok",
        "prompt_len": len(app.state.current_tokens),
        "prompt_ids": app.state.current_tokens,
    }

class VerifyRequest(BaseModel):
    draft_tokens: List[int]

@app.post("/verify/")
def verify(req: VerifyRequest):
    """
    Verify draft tokens against greedy predictions.
    Speculative decoding rule:
    - draft[0] must match our prediction from current context
    - draft[1] must match our prediction after accepting draft[0]
    - etc.
    """
    if DEBUG:
        print(f"\n=== VERIFY ===")
        print(f"Draft: {req.draft_tokens} = {repr(tokenizer.decode(req.draft_tokens))}")
    
    if app.state.kv_cache is None:
        return {"error": "No KV cache. Call /prefill first."}
    
    draft_tokens = req.draft_tokens
    if not draft_tokens:
        return {"accepted_prefix_len": 0, "preds": []}
    
    # Backup state
    original_kv = copy.deepcopy(app.state.kv_cache)
    original_tokens = app.state.current_tokens.copy()
    
    if DEBUG:
        print(f"Context: {len(original_tokens)} tokens")
        print(f"KV shape: {original_kv[0][0].shape}")
    
    # Verify token by token
    accepted = 0
    preds = []
    current_kv = original_kv
    
    for i, draft_token in enumerate(draft_tokens):
        if i == 0:
            # First token: use cached logits from prefill/previous verify
            if hasattr(app.state, 'prefill_logits') and app.state.prefill_logits is not None:
                logits = app.state.prefill_logits[0, -1, :]
                app.state.prefill_logits = None  # Use only once
            else:
                # Shouldn't happen, but fallback: recompute
                context_tensor = torch.tensor([original_tokens], dtype=torch.long).to(device)
                with torch.no_grad():
                    outputs = model(context_tensor, use_cache=False)
                    logits = outputs.logits[0, -1, :]
        else:
            # Subsequent tokens: use KV cache + previous accepted token
            prev_token_tensor = torch.tensor([[draft_tokens[i-1]]], dtype=torch.long).to(device)
            with torch.no_grad():
                outputs = model(prev_token_tensor, past_key_values=current_kv, use_cache=True)
                logits = outputs.logits[0, -1, :]
                current_kv = outputs.past_key_values
        
        # Get our greedy prediction
        our_pred = torch.argmax(logits).item()
        preds.append(our_pred)
        
        if DEBUG:
            match = "✓" if our_pred == draft_token else "✗"
            print(f"  [{i}] {match} pred={our_pred}({repr(tokenizer.decode([our_pred]))}) draft={draft_token}({repr(tokenizer.decode([draft_token]))})")
        
        if our_pred == draft_token:
            accepted += 1
        else:
            break  # Stop at first mismatch
    
    if DEBUG:
        print(f"Result: {accepted}/{len(draft_tokens)} accepted")
    
    # Update state with accepted tokens
    if accepted > 0:
        accepted_tensor = torch.tensor([draft_tokens[:accepted]], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(accepted_tensor, past_key_values=original_kv, use_cache=True)
            app.state.kv_cache = outputs.past_key_values
            app.state.current_tokens.extend(draft_tokens[:accepted])
            app.state.prefill_logits = outputs.logits  # Save for next verification
        
        if DEBUG:
            print(f"Updated: KV shape={app.state.kv_cache[0][0].shape}, tokens={len(app.state.current_tokens)}")
    else:
        # No changes
        app.state.kv_cache = original_kv
        app.state.current_tokens = original_tokens
        if DEBUG:
            print("State unchanged")
    
    return {
        "accepted_prefix_len": accepted,
        "preds": preds,
        "total_tokens": len(app.state.current_tokens)
    }

@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": str(device),
        "tokens": len(app.state.current_tokens) if app.state.current_tokens else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


# # main.py
# # Run with: uvicorn main:app --host 127.0.0.1 --port 8000

# from fastapi import FastAPI
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from pydantic import BaseModel
# from typing import List

# app = FastAPI(title="DGX Verifier (simulated)")

# # === Model and tokenizer shared globally ===
# MODEL_NAME = "distilgpt2"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Loading verifier model {MODEL_NAME} on {device} ...")
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Shared global state
# app.state.model = model
# app.state.tokenizer = tokenizer
# app.state.device = device
# app.state.kv_cache = None
# app.state.current_tokens = []

# # Enable debug mode
# DEBUG = True

# # === Prefill Endpoint ===
# class PrefillRequest(BaseModel):
#     prompt: str

# @app.post("/prefill/")
# def prefill(req: PrefillRequest):
#     """Run prefill and store KV cache"""
#     if DEBUG:
#         print(f"\n=== PREFILL REQUEST ===")
#         print(f"Prompt: {req.prompt}")
    
#     inputs = tokenizer(req.prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"].to(device)
    
#     if DEBUG:
#         print(f"Input IDs: {input_ids.squeeze(0).tolist()}")
#         print(f"Input shape: {input_ids.shape}")
    
#     with torch.no_grad():
#         outputs = model(input_ids, use_cache=True)
    
#     # Store KV cache and tokens
#     app.state.kv_cache = outputs.past_key_values
#     app.state.current_tokens = input_ids.squeeze(0).cpu().tolist()
    
#     if DEBUG:
#         print(f"KV cache layers: {len(outputs.past_key_values)}")
#         print(f"KV cache shape (layer 0): {outputs.past_key_values[0][0].shape}")
#         print(f"Stored tokens: {app.state.current_tokens}")
    
#     return {
#         "status": "ok",
#         "prompt_len": len(app.state.current_tokens),
#         "prompt_ids": app.state.current_tokens,
#     }

# # === Verify Endpoint ===
# class VerifyRequest(BaseModel):
#     draft_tokens: List[int]

# @app.post("/verify/")
# def verify(req: VerifyRequest):
#     """Verify draft tokens and update KV cache ONLY for accepted tokens"""
#     if DEBUG:
#         print(f"\n=== VERIFY REQUEST ===")
#         print(f"Draft tokens: {req.draft_tokens}")
#         print(f"Draft decoded: {repr(tokenizer.decode(req.draft_tokens))}")
    
#     if app.state.kv_cache is None:
#         return {"error": "No KV cache available. Call /prefill first."}
    
#     draft_tokens = req.draft_tokens
#     if len(draft_tokens) == 0:
#         return {"accepted_prefix_len": 0, "preds": []}
    
#     # CRITICAL: Save the original KV cache before verification
#     # Deep copy to prevent any modifications
#     import copy
#     original_kv_cache = app.state.kv_cache
#     original_kv_backup = copy.deepcopy(app.state.kv_cache) if app.state.kv_cache else None
    
#     # Convert draft tokens to tensor
#     draft_tensor = torch.tensor([draft_tokens], dtype=torch.long).to(device)
    
#     if DEBUG:
#         print(f"Current context length: {len(app.state.current_tokens)}")
#         print(f"KV cache shape before verify: {original_kv_cache[0][0].shape}")
    
#     # Run verification with cached KV (temporary outputs just for prediction)
#     with torch.no_grad():
#         temp_outputs = model(draft_tensor, past_key_values=original_kv_cache, use_cache=True)
#         logits = temp_outputs.logits
    
#     # Get predictions (argmax for greedy)
#     preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
    
#     if DEBUG:
#         print(f"Predictions: {preds}")
#         print(f"Predictions decoded: {[repr(tokenizer.decode([p])) for p in preds]}")
#         print(f"Draft decoded: {[repr(tokenizer.decode([d])) for d in draft_tokens]}")
    
#     # Count accepted tokens (consecutive matches)
#     accepted = 0
#     for i, (pred_token, draft_token) in enumerate(zip(preds, draft_tokens)):
#         if DEBUG:
#             match_str = "✓" if pred_token == draft_token else "✗"
#             print(f"  Position {i}: {match_str} pred={pred_token}({repr(tokenizer.decode([pred_token]))}) vs draft={draft_token}({repr(tokenizer.decode([draft_token]))})")
        
#         if pred_token == draft_token:
#             accepted += 1
#         else:
#             break
    
#     if DEBUG:
#         print(f"Accepted: {accepted}/{len(draft_tokens)}")
    
#     # CRITICAL FIX: Update KV cache ONLY with accepted tokens
#     if accepted > 0:
#         accepted_tensor = torch.tensor([draft_tokens[:accepted]], dtype=torch.long).to(device)
        
#         if DEBUG:
#             print(f"Updating KV cache with {accepted} accepted tokens...")
        
#         with torch.no_grad():
#             # Use the ORIGINAL cache, not the one from outputs above
#             new_outputs = model(accepted_tensor, past_key_values=original_kv_cache, use_cache=True)
#             app.state.kv_cache = new_outputs.past_key_values
#             app.state.current_tokens.extend(draft_tokens[:accepted])
        
#         if DEBUG:
#             print(f"KV cache updated. New shape: {app.state.kv_cache[0][0].shape}")
#             print(f"Total tokens now: {len(app.state.current_tokens)}")
#             print(f"Last few tokens: {app.state.current_tokens[-10:]}")
#     else:
#         # NO tokens accepted - keep the original KV cache unchanged
#         if DEBUG:
#             print(f"No tokens accepted - KV cache remains unchanged")
#         app.state.kv_cache = original_kv_cache
    
#     return {
#         "accepted_prefix_len": accepted,
#         "preds": preds,
#         "total_tokens": len(app.state.current_tokens)
#     }

# # === Health Check ===
# @app.get("/")
# def root():
#     return {
#         "status": "running",
#         "model": MODEL_NAME,
#         "device": str(device),
#         "tokens_generated": len(app.state.current_tokens) if app.state.current_tokens else 0
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)