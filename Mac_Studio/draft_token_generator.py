#!/usr/bin/env python3
"""
Mac: 4-bit quantized draft generator using llama.cpp
Install: CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python requests
Download model: wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Run: python client.py --dgx http://127.0.0.1:8000 --prompt "Count from 1 to 10"
"""

import argparse
import time
import requests
import numpy as np
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("--dgx", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
parser.add_argument("--draft_n", type=int, default=4)
parser.add_argument("--max_tokens", type=int, default=80)
parser.add_argument("--prompt", type=str, default="Count from 1 to 10")
args = parser.parse_args()

print(f"Loading 4-bit model: {args.model}")
llm = Llama(model_path=args.model, n_ctx=2048, n_gpu_layers=99, verbose=False, logits_all=True)
print("✓ Model loaded\n")

# Prefill
print(f"Prompt: {args.prompt}")
r = requests.post(f"{args.dgx}/prefill/", json={"prompt": args.prompt})
r.raise_for_status()
prompt_len = r.json()["prompt_len"]
current_tokens = r.json()["prompt_ids"]
print(f"✓ Prefill done ({prompt_len} tokens)\n")

# Speculative decoding loop
start = time.time()
stats = {"drafted": 0, "accepted": 0, "iters": 0}

while len(current_tokens) < args.max_tokens:
    stats["iters"] += 1
    print(f"Iter {stats['iters']}: {len(current_tokens)}/{args.max_tokens} tokens")
    
    # Draft tokens (greedy) - generate from current context
    draft_tokens = []
    
    # CRITICAL: Generate tokens continuing from current_tokens, not from scratch
    prompt_text = llm.detokenize(current_tokens).decode('utf-8', errors='ignore')
    
    print(f"  Current context preview: ...{repr(prompt_text[-50:])}")
    
    # Use the model's generate method with temperature=0 for greedy
    output = llm(
        prompt_text,
        max_tokens=args.draft_n,
        temperature=0.0,  # Greedy
        echo=False
    )
    
    generated_text = output['choices'][0]['text']
    print(f"  Generated text: {repr(generated_text)[:80]}")
    
    # Tokenize the generated text
    draft_tokens = llm.tokenize(generated_text.encode('utf-8'))
    
    # Filter out any leading BOS tokens or zeros from tokenization
    draft_tokens = [t for t in draft_tokens if t not in [0, 1, 2]]
    
    if not draft_tokens:
        print("  ⚠️ No valid draft tokens generated (all special tokens)")
        break
    
    stats["drafted"] += len(draft_tokens)
    draft_text = llm.detokenize(draft_tokens).decode('utf-8', errors='ignore')
    print(f"  Draft tokens: {draft_tokens}")
    print(f"  Draft text: {repr(draft_text)[:50]}")
    
    # Verify
    r = requests.post(f"{args.dgx}/verify/", json={"draft_tokens": draft_tokens})
    r.raise_for_status()
    accepted = r.json()["accepted_prefix_len"]
    
    if accepted > 0:
        current_tokens.extend(draft_tokens[:accepted])
        stats["accepted"] += accepted
        accepted_text = llm.detokenize(draft_tokens[:accepted]).decode('utf-8', errors='ignore')
        print(f" Correct. {accepted}/{len(draft_tokens)} accepted: {repr(accepted_text)[:50]}\n")
    else:
        # All draft tokens rejected - get server's prediction and use that
        print(" Wrong. All draft tokens rejected. Using server's prediction...")
        preds = r.json().get("preds", [])
        if preds and len(preds) > 0:
            # Take the first predicted token from server
            server_token = preds[0]
            print(f"  Server predicted token: {server_token}")
            
            # Skip if it's a special token
            if server_token in [0, 1, 2]:
                print(f"  ⚠️ Server returned special token {server_token}, stopping")
                break
                
            current_tokens.append(server_token)
            stats["accepted"] += 1
            stats["drafted"] += 1
            server_text = llm.detokenize([server_token]).decode('utf-8', errors='ignore')
            print(f" Great, Using server token: {repr(server_text)[:50]}\n")
        else:
            print(" No server prediction available. Stopping.\n")
            break

total_time = time.time() - start
output = llm.detokenize(current_tokens).decode('utf-8', errors='ignore')

# Calculate metrics
new_tokens = len(current_tokens) - prompt_len
tokens_per_sec = new_tokens / total_time if total_time > 0 else 0
acceptance_rate = (stats['accepted'] / stats['drafted'] * 100) if stats['drafted'] > 0 else 0

# Estimate baseline (sequential) throughput
# Assume sequential generation would take 1 verification per token
avg_verify_time = total_time / stats['iters'] if stats['iters'] > 0 else 0
estimated_sequential_time = new_tokens * avg_verify_time
estimated_sequential_tps = new_tokens / estimated_sequential_time if estimated_sequential_time > 0 else 0
speedup = tokens_per_sec / estimated_sequential_tps if estimated_sequential_tps > 0 else 1.0

print("="*60)
print(f"\n📝 Generated Text:")
print(f"{'='*60}")
print(output)
print(f"{'='*60}\n")



print("="*60)
print("SPECULATIVE DECODING RESULTS")
print("="*60)
print(f"\n📊 Generation Stats:")
print(f"   Prompt tokens:        {prompt_len}")
print(f"   Generated tokens:     {new_tokens}")
print(f"   Total tokens:         {len(current_tokens)}")
print(f"   Iterations:           {stats['iters']}")

print(f"\n⚡ Performance:")
print(f"   Total time:           {total_time:.2f}s")
print(f"   Throughput:           {tokens_per_sec:.1f} tok/s")

print(f"\n🎯 Speculative Decoding Efficiency:")
print(f"   Tokens drafted:       {stats['drafted']}")
print(f"   Tokens accepted:      {stats['accepted']}")
print(f"   Acceptance rate:      {acceptance_rate:.1f}%")
print(f"   Avg tokens/iteration: {new_tokens/stats['iters']:.2f}")

print(f"\n🚀 Speedup Analysis:")
print(f"   Speculative:          {tokens_per_sec:.1f} tok/s")
print(f"   Est. Sequential:      {estimated_sequential_tps:.1f} tok/s")
print(f"   Speedup:              {speedup:.2f}x faster")
