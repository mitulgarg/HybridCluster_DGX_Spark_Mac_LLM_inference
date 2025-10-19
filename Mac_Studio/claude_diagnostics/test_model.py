#!/usr/bin/env python3
"""
Test script to verify both draft and verifier models generate the same tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
PROMPT = "Hello please think about what time means and answer"

print(f"Testing model: {MODEL_NAME}")
print(f"Prompt: {PROMPT}\n")

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Encode prompt
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
print(f"Input IDs: {input_ids.squeeze(0).tolist()}")
print(f"Input shape: {input_ids.shape}\n")

# Method 1: Using model.generate (what draft uses)
print("="*60)
print("METHOD 1: model.generate (Draft Model Method)")
print("="*60)
with torch.no_grad():
    output_generate = model.generate(
        input_ids,
        max_new_tokens=4,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.eos_token_id,
    )
generated_tokens = output_generate[0][len(input_ids[0]):].cpu().tolist()
print(f"Generated tokens: {generated_tokens}")
print(f"Generated text: {repr(tokenizer.decode(generated_tokens))}")

# Method 2: Using forward pass + argmax (what verifier uses)
print("\n" + "="*60)
print("METHOD 2: forward + argmax (Verifier Method)")
print("="*60)
with torch.no_grad():
    # First, get the KV cache from prefill
    outputs_prefill = model(input_ids, use_cache=True)
    kv_cache = outputs_prefill.past_key_values
    
    # Now predict next tokens one by one
    predicted_tokens = []
    current_kv = kv_cache
    
    for i in range(4):
        if i == 0:
            # First token: predict directly from KV cache
            logits = outputs_prefill.logits
            next_token = torch.argmax(logits[0, -1, :]).item()
        else:
            # Subsequent tokens: use previous token + KV cache
            prev_token_tensor = torch.tensor([[predicted_tokens[-1]]], dtype=torch.long).to(device)
            outputs_next = model(prev_token_tensor, past_key_values=current_kv, use_cache=True)
            logits = outputs_next.logits
            next_token = torch.argmax(logits[0, -1, :]).item()
            current_kv = outputs_next.past_key_values
        
        predicted_tokens.append(next_token)
        print(f"  Step {i+1}: token={next_token} ({repr(tokenizer.decode([next_token]))})")

print(f"\nPredicted tokens: {predicted_tokens}")
print(f"Predicted text: {repr(tokenizer.decode(predicted_tokens))}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Generate method:  {generated_tokens}")
print(f"Forward method:   {predicted_tokens}")
print(f"Match: {generated_tokens == predicted_tokens}")

if generated_tokens != predicted_tokens:
    print("\n⚠️  MISMATCH DETECTED!")
    print("This explains why speculative decoding is failing.")
    print("\nToken-by-token comparison:")
    for i, (gen, pred) in enumerate(zip(generated_tokens, predicted_tokens)):
        match = "✓" if gen == pred else "✗"
        gen_text = tokenizer.decode([gen])
        pred_text = tokenizer.decode([pred])
        print(f"  Position {i}: {match} generate={gen}({repr(gen_text)}) forward={pred}({repr(pred_text)})")
else:
    print("\n✓ Both methods produce identical tokens!")
    print("The models should work correctly for speculative decoding.")

# Additional test: Check what the model actually wants to generate
print("\n" + "="*60)
print("NEXT TOKEN PROBABILITIES (Top 5)")
print("="*60)
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]  # Last token logits
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)
    
    print("Most likely next tokens:")
    for prob, idx in zip(top_probs, top_indices):
        token_text = tokenizer.decode([idx.item()])
        print(f"  Token {idx.item()}: {repr(token_text)} (prob: {prob.item():.4f})")