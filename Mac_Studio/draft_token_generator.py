# draft_token_generator.py
# Fixed version with proper greedy decoding and attention mask

import argparse
import time
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dgx", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="distilgpt2")
parser.add_argument("--draft_n", type=int, default=4)
parser.add_argument("--max_tokens", type=int, default=100)
parser.add_argument("--prompt", type=str, default="Explain quantum entanglement simply.")
args = parser.parse_args()

DGX_URL = args.dgx.rstrip("/")
DRAFT_N = args.draft_n
MAX_TOKENS = args.max_tokens
PROMPT = args.prompt

print(f"Loading drafter model {args.model} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

stats = {
    "total_drafted": 0,
    "total_accepted": 0,
    "iterations": 0,
    "draft_time": 0.0,
    "verify_time": 0.0,
    "acceptance_by_iter": [],
}

print(f"\n{'='*60}")
print(f"PREFILL PHASE")
print(f"{'='*60}")
print(f"Prompt: {PROMPT}")

r = requests.post(f"{DGX_URL}/prefill/", json={"prompt": PROMPT})
if r.status_code != 200:
    print(f"❌ Prefill failed: {r.text}")
    raise SystemExit(1)

prefill_resp = r.json()
prompt_len = prefill_resp.get("prompt_len", 0)
print(f"✓ DGX prefill ready. Prompt length = {prompt_len} tokens")

tokens = tokenizer.encode(PROMPT)
committed = len(tokens)

print(f"\n{'='*60}")
print(f"SPECULATIVE DECODING LOOP")
print(f"{'='*60}")
print(f"Target: {MAX_TOKENS} tokens | Draft window: {DRAFT_N}\n")

start_time = time.time()
iteration = 0

while len(tokens) < MAX_TOKENS:
    iteration += 1
    iter_start = time.time()
    
    print(f"\n--- Iteration {iteration} ---")
    print(f"Current: {len(tokens)}/{MAX_TOKENS} tokens")
    
    # Generate draft tokens - GREEDY with proper attention mask
    draft_start = time.time()
    input_ids = torch.tensor([tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)  # Proper attention mask
    
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Add attention mask
            max_new_tokens=DRAFT_N,
            do_sample=False,  # GREEDY
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = out[0].cpu().tolist()[len(tokens):]
    draft_time = time.time() - draft_start
    stats["draft_time"] += draft_time
    
    if len(generated) == 0:
        print("⚠️  No more tokens. Stopping.")
        break
    
    stats["total_drafted"] += len(generated)
    draft_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"📝 Drafted {len(generated)} tokens: {repr(draft_text)[:60]}")
    print(f"   Token IDs: {generated}")
    
    # Verify with DGX
    verify_start = time.time()
    r = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": generated})
    verify_time = time.time() - verify_start
    stats["verify_time"] += verify_time
    
    if r.status_code != 200:
        print(f"❌ Verify failed: {r.text}")
        break
    
    resp = r.json()
    accepted = resp.get("accepted_prefix_len", 0)
    preds = resp.get("preds", [])
    
    if accepted > 0:
        accepted_text = tokenizer.decode(generated[:accepted], skip_special_tokens=True)
        tokens.extend(generated[:accepted])
        stats["total_accepted"] += accepted
        stats["acceptance_by_iter"].append(accepted)
        
        acceptance_rate = (accepted / len(generated)) * 100
        print(f"✅ Accepted {accepted}/{len(generated)} ({acceptance_rate:.0f}%): {repr(accepted_text)[:60]}")
        print(f"   Output so far: ...{repr(tokenizer.decode(tokens[-20:]))}")
        committed = len(tokens)
    else:
        print(f"❌ Rejected all {len(generated)} tokens")
        stats["acceptance_by_iter"].append(0)
        
        # Show why rejected
        if len(preds) > 0:
            print(f"   Expected: {preds[:3]}... Got: {generated[:3]}...")
        
        # Fallback: single token
        print("   Trying fallback (1 token)...")
        with torch.no_grad():
            out2 = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        gen1 = out2[0].cpu().tolist()[len(tokens):]
        if not gen1:
            print("⚠️  Cannot generate. Stopping.")
            break
        
        stats["total_drafted"] += 1
        r2 = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": gen1})
        resp2 = r2.json()
        accepted2 = resp2.get("accepted_prefix_len", 0)
        
        if accepted2 > 0:
            tokens.extend(gen1)
            stats["total_accepted"] += 1
            fallback_text = tokenizer.decode(gen1, skip_special_tokens=True)
            print(f"   ✓ Fallback accepted: {repr(fallback_text)}")
            committed = len(tokens)
        else:
            print("   ❌ Fallback rejected. Aborting.")
            break
    
    stats["iterations"] += 1
    iter_time = time.time() - iter_start
    print(f"⏱️  {iter_time:.2f}s (draft: {draft_time:.2f}s, verify: {verify_time:.2f}s)")

end_time = time.time()
total_time = end_time - start_time
new_tokens = len(tokens) - prompt_len

# Statistics
print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\n📊 Token Statistics:")
print(f"   Prompt tokens:    {prompt_len}")
print(f"   Generated tokens: {new_tokens}")
print(f"   Total tokens:     {len(tokens)}")
print(f"\n⚡ Performance:")
print(f"   Total time:       {total_time:.2f}s")
if new_tokens > 0:
    print(f"   Tokens/second:    {new_tokens/total_time:.1f}")
print(f"   Draft time:       {stats['draft_time']:.2f}s ({stats['draft_time']/total_time*100:.0f}%)")
print(f"   Verify time:      {stats['verify_time']:.2f}s ({stats['verify_time']/total_time*100:.0f}%)")

if stats['total_drafted'] > 0:
    print(f"\n🎯 Speculative Decoding:")
    print(f"   Iterations:       {stats['iterations']}")
    print(f"   Tokens drafted:   {stats['total_drafted']}")
    print(f"   Tokens accepted:  {stats['total_accepted']}")
    acceptance_rate = (stats['total_accepted'] / stats['total_drafted']) * 100
    print(f"   Acceptance rate:  {acceptance_rate:.1f}%")
    
    if new_tokens > 0 and stats['iterations'] > 0:
        avg_verify_time = stats['verify_time'] / stats['iterations']
        estimated_regular = new_tokens * avg_verify_time
        speedup = estimated_regular / total_time if total_time > 0 else 1.0
        print(f"\n🚀 Speedup Estimate:")
        print(f"   Regular: {estimated_regular:.2f}s vs Speculative: {total_time:.2f}s")
        print(f"   Speedup: {speedup:.2f}x")

print(f"\n📝 Final Output:")
print(f"{'='*60}")
output_text = tokenizer.decode(tokens)
print(output_text)
print(f"{'='*60}")

if stats['acceptance_by_iter']:
    print(f"\n📈 Acceptance per iteration: {stats['acceptance_by_iter']}")