#!/usr/bin/env python3
"""
Diagnostic script to find why draft and verifier predict different tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME_OLD = "distilgpt2"
MODEL_NAME = "gpt2"  # Alternative model to test

PROMPT = "Hello please think about what time means and answer"

print(f"Diagnosing prediction mismatch for: {MODEL_NAME}")
print(f"Prompt: {PROMPT}\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Encode prompt
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
print(f"Input IDs: {input_ids.squeeze(0).tolist()}")
print(f"Input shape: {input_ids.shape}")

print("\n" + "="*70)
print("TEST 1: Using model.generate() - What DRAFT does")
print("="*70)

attention_mask = torch.ones_like(input_ids)
with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

draft_token = output[0][-1].item()
draft_text = tokenizer.decode([draft_token])
print(f"Draft predicts: token={draft_token} ({repr(draft_text)})")

print("\n" + "="*70)
print("TEST 2: Using forward pass + KV cache - What VERIFIER does")
print("="*70)

# Step 1: Prefill (get KV cache)
with torch.no_grad():
    prefill_outputs = model(input_ids, use_cache=True)
    kv_cache = prefill_outputs.past_key_values
    logits_from_prefill = prefill_outputs.logits

# Get prediction from prefill logits (last token)
verifier_token_from_prefill = torch.argmax(logits_from_prefill[0, -1, :]).item()
verifier_text = tokenizer.decode([verifier_token_from_prefill])
print(f"Verifier predicts from prefill: token={verifier_token_from_prefill} ({repr(verifier_text)})")

print(f"\nKV cache shape (layer 0): {kv_cache[0][0].shape}")

# Step 2: Now verify what happens if we pass draft token through with KV cache
print("\n" + "="*70)
print("TEST 3: Simulating verification of draft token")
print("="*70)

draft_tensor = torch.tensor([[draft_token]], dtype=torch.long).to(device)
print(f"Verifying draft token: {draft_token} ({repr(draft_text)})")

with torch.no_grad():
    verify_outputs = model(draft_tensor, past_key_values=kv_cache, use_cache=True)
    verify_logits = verify_outputs.logits
    
predicted_next = torch.argmax(verify_logits[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_next])

print(f"If we accept draft {draft_token}, next token would be: {predicted_next} ({repr(predicted_text)})")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

if draft_token == verifier_token_from_prefill:
    print("✅ MATCH: Draft and verifier agree on next token")
else:
    print("❌ MISMATCH: Draft and verifier disagree!")
    print(f"   Draft says:    token {draft_token} ({repr(draft_text)})")
    print(f"   Verifier says: token {verifier_token_from_prefill} ({repr(verifier_text)})")
    
    print("\n" + "="*70)
    print("DEEP DIVE: Checking logit values")
    print("="*70)
    
    # Get logits from both methods
    with torch.no_grad():
        # Method 1: Direct forward pass
        direct_outputs = model(input_ids)
        direct_logits = direct_outputs.logits[0, -1, :]
        
        # Method 2: Forward pass with use_cache
        cached_outputs = model(input_ids, use_cache=True)
        cached_logits = cached_outputs.logits[0, -1, :]
    
    print(f"Logits shape: {direct_logits.shape}")
    print(f"\nTop 5 predictions from direct forward pass:")
    top_direct = torch.topk(direct_logits, 5)
    for prob, idx in zip(top_direct.values, top_direct.indices):
        token_text = tokenizer.decode([idx.item()])
        print(f"  Token {idx.item()}: {repr(token_text)} (logit: {prob.item():.4f})")
    
    print(f"\nTop 5 predictions from cached forward pass:")
    top_cached = torch.topk(cached_logits, 5)
    for prob, idx in zip(top_cached.values, top_cached.indices):
        token_text = tokenizer.decode([idx.item()])
        print(f"  Token {idx.item()}: {repr(token_text)} (logit: {prob.item():.4f})")
    
    print(f"\nLogit difference (max absolute): {torch.max(torch.abs(direct_logits - cached_logits)).item():.6f}")
    
    if torch.max(torch.abs(direct_logits - cached_logits)).item() < 1e-5:
        print("✓ Logits are essentially identical")
    else:
        print("✗ Logits differ significantly!")

print("\n" + "="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

# Check if model.generate uses different settings
print("Checking model.generate internals...")
print(f"Model config:")
print(f"  - vocab_size: {model.config.vocab_size}")
print(f"  - eos_token_id: {model.config.eos_token_id}")
print(f"  - pad_token_id: {tokenizer.pad_token_id}")
print(f"  - bos_token_id: {model.config.bos_token_id}")

# The issue might be in how generate handles attention
print("\nTesting with explicit attention patterns...")
print("This might reveal if padding/masking causes differences.")