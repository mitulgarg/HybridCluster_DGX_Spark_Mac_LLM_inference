# draft_token_generator.py
# Usage: python draft_token_generator.py --dgx http://127.0.0.1:8000 --draft_n 4 --max_tokens 80 --prompt "Hello please think about what time means"

import argparse
import time
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CLI arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--dgx", default="http://127.0.0.1:8000", help="DGX verifier server URL")
parser.add_argument("--model", default="distilgpt2", help="Drafter model to use locally (small for CPU)")
parser.add_argument("--draft_n", type=int, default=4, help="Speculative window size (draft tokens per batch)")
parser.add_argument("--max_tokens", type=int, default=100, help="Max total tokens including prompt")
parser.add_argument("--prompt", type=str, default="Explain quantum entanglement simply.", help="Prompt")
parser.add_argument("--simulate", action="store_true", help="Force acceptance to simulate speculative decoding")
args = parser.parse_args()

# --- Configuration ---
DGX_URL = args.dgx.rstrip("/")
DRAFT_N = args.draft_n
MAX_TOKENS = args.max_tokens
PROMPT = args.prompt
SIMULATE = args.simulate

print(f"Loading drafter model {args.model} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Step 1: Send prefill to DGX ---
print("Sending prefill to DGX ...")
r = requests.post(f"{DGX_URL}/prefill", json={"prompt": PROMPT})
if r.status_code != 200:
    print("Prefill failed:", r.text)
    raise SystemExit(1)
prefill_resp = r.json()
prompt_len = prefill_resp.get("prompt_len", None)
print(f"DGX prefill ready. Prompt length = {prompt_len}")

# --- Step 2: Initialize token list ---
tokens = tokenizer.encode(PROMPT)
committed = len(tokens)

print("Starting speculative decode loop ...")
start_time = time.time()
dgx_only_time = 0.0  # For throughput comparison

while len(tokens) < MAX_TOKENS:
    iter_start = time.time()

    input_ids = torch.tensor([tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)

    # --- Generate draft tokens (greedy) ---
    out = model.generate(
        input_ids,
        max_new_tokens=DRAFT_N,
        do_sample=False,  # greedy
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
    )
    generated = out[0].cpu().tolist()[len(tokens):]
    if len(generated) == 0:
        print("\nNo more tokens generated. Stopping.")
        break

    # --- Send draft to DGX for verification ---
    if SIMULATE:
        # For simulation: accept all draft tokens
        accepted = len(generated)
        dgx_time = 0.0
    else:
        verify_start = time.time()
        r = requests.post(f"{DGX_URL}/verify", json={"draft_tokens": generated})
        dgx_time = time.time() - verify_start
        dgx_only_time += dgx_time
        if r.status_code != 200:
            print("Verify failed:", r.text)
            break
        resp = r.json()
        accepted = resp.get("accepted_prefix_len", 0)

    print("\nDraft tokens:", generated)
    print("Accepted tokens:", generated[:accepted])

    # --- Commit accepted tokens ---
    if accepted > 0:
        tokens.extend(generated[:accepted])
        committed_text = tokenizer.decode(tokens[committed:])
        print(committed_text, end="", flush=True)
        committed = len(tokens)
    else:
        print("\nNo tokens accepted; retrying with draft_n=1 once.")
        out2 = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        gen1 = out2[0].cpu().tolist()[len(tokens):]
        if not gen1:
            print("Drafter could not produce a token. Stopping.")
            break
        if SIMULATE:
            accepted2 = len(gen1)
            dgx_time = 0.0
        else:
            verify_start = time.time()
            r2 = requests.post(f"{DGX_URL}/verify", json={"draft_tokens": gen1})
            dgx_time = time.time() - verify_start
            dgx_only_time += dgx_time
            resp2 = r2.json()
            accepted2 = resp2.get("accepted_prefix_len", 0)
        if accepted2 > 0:
            tokens.extend(gen1[:accepted2])
            committed_text = tokenizer.decode(tokens[committed:])
            print(committed_text, end="", flush=True)
            committed = len(tokens)
        else:
            print("Still not accepted. Aborting.")
            break

    iter_end = time.time()
    print(f"Iteration took {iter_end - iter_start:.3f}s, DGX time {dgx_time:.3f}s, accepted {accepted}/{len(generated)} tokens")
    time.sleep(0.05)

end_time = time.time()
total_time = end_time - start_time
print("\n\nGeneration finished. Total tokens:", len(tokens), "Duration:", total_time, "s")
if SIMULATE and len(tokens) > len(tokenizer.encode(PROMPT)):
    dgx_only_estimate = total_time + dgx_only_time
    speedup = dgx_only_estimate / total_time
    print(f"Estimated DGX-only time: {dgx_only_estimate:.3f}s, Speculative speedup: {speedup:.2f}x")

print("\nFinal output:\n", tokenizer.decode(tokens))
