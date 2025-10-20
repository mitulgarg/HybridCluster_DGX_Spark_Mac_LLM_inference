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
parser.add_argument("--model", default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
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
        print(f"  ✅ {accepted}/{len(draft_tokens)} accepted: {repr(accepted_text)[:50]}\n")
    else:
        # All draft tokens rejected - get server's prediction and use that
        print("  ❌ All draft tokens rejected. Using server's prediction...")
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
            print(f"  ✅ Using server token: {repr(server_text)[:50]}\n")
        else:
            print("  ❌ No server prediction available. Stopping.\n")
            break

total_time = time.time() - start
output = llm.detokenize(current_tokens).decode('utf-8', errors='ignore')

print("="*60)
print(f"Generated: {len(current_tokens) - prompt_len} tokens in {total_time:.2f}s")
print(f"Speed: {(len(current_tokens) - prompt_len)/total_time:.1f} tok/s")
print(f"Acceptance: {stats['accepted']}/{stats['drafted']} ({100*stats['accepted']/max(1,stats['drafted']):.0f}%)")
print("="*60)
print(f"\nToken IDs: {current_tokens}")
print(f"\nFull output:\n{output}\n")


# #!/usr/bin/env python3
# """
# Mac: 4-bit quantized draft generator using llama.cpp
# Install: CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python requests
# Download model: wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# Run: python client.py --dgx http://127.0.0.1:8000 --prompt "Count from 1 to 10"
# """

# import argparse
# import time
# import requests
# import numpy as np
# from llama_cpp import Llama

# parser = argparse.ArgumentParser()
# parser.add_argument("--dgx", default="http://127.0.0.1:8000")
# parser.add_argument("--model", default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
# parser.add_argument("--draft_n", type=int, default=4)
# parser.add_argument("--max_tokens", type=int, default=80)
# parser.add_argument("--prompt", type=str, default="Count from 1 to 10")
# args = parser.parse_args()

# print(f"Loading 4-bit model: {args.model}")
# llm = Llama(model_path=args.model, n_ctx=2048, n_gpu_layers=99, verbose=False)
# print("✓ Model loaded\n")

# # Prefill
# print(f"Prompt: {args.prompt}")
# r = requests.post(f"{args.dgx}/prefill/", json={"prompt": args.prompt})
# r.raise_for_status()
# prompt_len = r.json()["prompt_len"]
# current_tokens = r.json()["prompt_ids"]
# print(f"✓ Prefill done ({prompt_len} tokens)\n")

# # Speculative decoding loop
# start = time.time()
# stats = {"drafted": 0, "accepted": 0, "iters": 0}

# while len(current_tokens) < args.max_tokens:
#     stats["iters"] += 1
#     print(f"Iter {stats['iters']}: {len(current_tokens)}/{args.max_tokens} tokens")
    
#     # Draft tokens (greedy)
#     llm.reset()
#     llm.eval(current_tokens)
#     draft_tokens = []
#     for _ in range(args.draft_n):
#         # FIX: Access logits as numpy array from the list
#         logits = np.array(llm.eval_logits[-1])
#         next_token = int(logits.argmax())
#         draft_tokens.append(next_token)
#         llm.eval([next_token])
    
#     stats["drafted"] += len(draft_tokens)
#     draft_text = llm.detokenize(draft_tokens).decode('utf-8', errors='ignore')
#     print(f"  Draft tokens: {draft_tokens}")
#     print(f"  Draft text: {repr(draft_text)[:50]}")
    
#     # Verify
#     r = requests.post(f"{args.dgx}/verify/", json={"draft_tokens": draft_tokens})
#     r.raise_for_status()
#     accepted = r.json()["accepted_prefix_len"]
    
#     if accepted > 0:
#         current_tokens.extend(draft_tokens[:accepted])
#         stats["accepted"] += accepted
#         print(f"  ✅ {accepted}/{len(draft_tokens)} accepted\n")
#     else:
#         # Fallback: single token
#         llm.reset()
#         llm.eval(current_tokens)
#         logits = np.array(llm.eval_logits[-1])
#         fallback = int(logits.argmax())
#         r = requests.post(f"{args.dgx}/verify/", json={"draft_tokens": [fallback]})
#         if r.json()["accepted_prefix_len"] > 0:
#             current_tokens.append(fallback)
#             stats["accepted"] += 1
#             stats["drafted"] += 1
#             print(f"  ✅ Fallback accepted\n")
#         else:
#             print("  ❌ Rejected, stopping\n")
#             break

# total_time = time.time() - start
# output = llm.detokenize(current_tokens).decode('utf-8', errors='ignore')

# print("="*60)
# print(f"Generated: {len(current_tokens) - prompt_len} tokens in {total_time:.2f}s")
# print(f"Speed: {(len(current_tokens) - prompt_len)/total_time:.1f} tok/s")
# print(f"Acceptance: {stats['accepted']}/{stats['drafted']} ({100*stats['accepted']/max(1,stats['drafted']):.0f}%)")
# print("="*60)
# print(f"\n{output}\n")

# #!/usr/bin/env python3
# """
# Speculative Decoding Client (Apple Silicon)
# Uses 4-bit GGUF quantized model via ctransformers for draft generation.
# """

# import argparse, time, requests, torch
# # -------------------------------------------------------------------
# # FIX 1: Import from the "transformers" wrapper module
# # -------------------------------------------------------------------
# from ctransformers.transformers import AutoModelForCausalLM
# from transformers import AutoTokenizer

# # ----------------------------
# # CLI ARGS
# # ----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--dgx", default="http://127.0.0.1:8000")
# parser.add_argument("--draft_n", type=int, default=4)
# parser.add_argument("--max_tokens", type=int, default=80)
# parser.add_argument("--prompt", type=str, default="Count from 1 to 10 goes like")
# args = parser.parse_args()

# DGX_URL = args.dgx.rstrip("/")
# DRAFT_N = args.draft_n
# MAX_TOKENS = args.max_tokens
# PROMPT = args.prompt

# # ----------------------------
# # MODEL CONFIG (Quantized)
# # ----------------------------
# MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# TOKENIZER_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_TYPE = "llama"
# device = "mps" # Apple Silicon GPU

# print("\n============================================================")
# print("LOADING DRAFT MODEL (Apple Silicon 4-bit)")
# print("============================================================")
# print(f"Model: {MODEL_REPO}/{MODEL_FILE}")

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_REPO,
#     model_file=MODEL_FILE,
#     model_type=MODEL_TYPE,
#     gpu_layers=60,           # offload most layers to M-series GPU
#     trust_remote_code=True
# ).to(device) # We can send the wrapper to the device
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
# print("✓ Model and tokenizer loaded successfully")

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # ----------------------------
# # PREFILL PHASE
# # ----------------------------
# print("\n============================================================")
# print("PREFILL PHASE")
# print("============================================================")
# print(f"Prompt: {PROMPT}")

# r = requests.post(f"{DGX_URL}/prefill/", json={"prompt": PROMPT})
# r.raise_for_status()
# prompt_len = r.json().get("prompt_len", 0)
# print(f"✓ DGX prefill ready. Prompt length = {prompt_len} tokens")

# tokens = tokenizer.encode(PROMPT)

# # -------------------------------------------------------------------
# # FIX 2: Use Hugging Face wrapper params: `max_new_tokens`
# #        and use GREEDY decoding (`do_sample=False`)
# # -------------------------------------------------------------------
# GEN_PARAMS = dict(
#     max_new_tokens=DRAFT_N,
#     do_sample=False,
#     pad_token_id=tokenizer.eos_token_id
# )

# print("\n============================================================")
# print("SPECULATIVE DECODING LOOP")
# print("============================================================")
# print(f"Target: {MAX_TOKENS} tokens | Draft window: {DRAFT_N}\n")

# start_time = time.time()
# stats = {"draft_time":0.0, "verify_time":0.0, "accepted":0, "drafted":0, "iters":0}

# while len(tokens) < MAX_TOKENS:
#     iter_start = time.time()
#     print(f"\n--- Iteration {stats['iters']+1} --- ({len(tokens)}/{MAX_TOKENS})")

#     # -------------------------------------------------------------------
#     # FIX 3: Pass a torch.Tensor to the wrapper's generate function
#     # -------------------------------------------------------------------
#     t0 = time.time()
#     # Create a tensor on the correct device
#     input_ids = torch.tensor([tokens]).to(device)
#     attention_mask = torch.ones_like(input_ids)

#     with torch.no_grad():
#         out = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             **GEN_PARAMS
#         )
    
#     # Get just the new tokens
#     draft_tokens = out[0].cpu().tolist()[len(tokens):]
#     stats["draft_time"] += time.time() - t0

#     if not draft_tokens:
#         print("⚠️  No draft tokens. Stopping.")
#         break

#     stats["drafted"] += len(draft_tokens)
#     draft_text = tokenizer.decode(draft_tokens, skip_special_tokens=True)
#     print(f"📝 Drafted {len(draft_tokens)} tokens: {repr(draft_text)[:60]}")
    
#     # Verify on DGX
#     v0 = time.time()
#     r = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": draft_tokens})
#     r.raise_for_status() 
#     verify_resp = r.json()
#     stats["verify_time"] += time.time() - v0

#     accepted = verify_resp.get("accepted_prefix_len", 0)
#     if accepted > 0:
#         accepted_text = tokenizer.decode(draft_tokens[:accepted], skip_special_tokens=True)
#         tokens.extend(draft_tokens[:accepted])
#         stats["accepted"] += accepted
#         print(f"✅ Accepted {accepted}/{len(draft_tokens)} → {repr(accepted_text)[:60]}")
#     else:
#         print(f"❌ Rejected all {len(draft_tokens)} tokens. Trying 1 token fallback...")
        
#         # --- Fallback Logic ---
#         fallback_params = dict(
#             max_new_tokens=1, 
#             do_sample=False, 
#             pad_token_id=tokenizer.eos_token_id
#         )
#         t_fallback = time.time()
#         with torch.no_grad():
#             fallback_out = model.generate(
#                 input_ids, 
#                 attention_mask=attention_mask, 
#                 **fallback_params
#             )
#         fallback_token = fallback_out[0].cpu().tolist()[len(tokens):]
#         stats["draft_time"] += time.time() - t_fallback

#         if not fallback_token:
#             print("⚠️  Fallback failed. Stopping.")
#             break
            
#         stats["drafted"] += 1
        
#         # Verify the single fallback token
#         v_fallback = time.time()
#         r_fallback = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": fallback_token})
#         stats["verify_time"] += time.time() - v_fallback
        
#         if r_fallback.status_code == 200 and r_fallback.json().get("accepted_prefix_len", 0) > 0:
#             print(f"   ✓ Fallback accepted: {repr(tokenizer.decode(fallback_token, skip_special_tokens=True))}")
#             tokens.extend(fallback_token)
#             stats["accepted"] += 1
#         else:
#             print("   ❌ Fallback token also rejected. Aborting.")
#             break

#     stats["iters"] += 1
#     if len(tokens) >= MAX_TOKENS:
#         break

# # ----------------------------
# # SUMMARY
# # ----------------------------
# total_time = time.time() - start_time
# gen_text = tokenizer.decode(tokens, skip_special_tokens=True)
# total_gen = len(tokens) - prompt_len

# print("\n============================================================")
# print("GENERATION COMPLETE")
# print("============================================================")
# print(f"Tokens generated: {total_gen}")
# print(f"Tokens accepted: {stats['accepted']}/{stats['drafted']} "
#       f"({100*stats['accepted']/max(1,stats['drafted']):.1f}%)")
# print(f"Total time: {total_time:.2f}s, Tokens/sec: {total_gen/max(1,total_time):.2f}")
# print(f"Draft time: {stats['draft_time']:.2f}s, Verify time: {stats['verify_time']:.2f}s")
# print(f"\nFinal text:\n{'='*60}\n{gen_text}\n{'='*60}")

# #!/usr/bin/env python3
# """
# Speculative Decoding Client (Apple Silicon)
# Uses 4-bit GGUF quantized model via ctransformers for draft generation.
# """

# import argparse, time, requests
# from ctransformers import AutoModelForCausalLM
# from transformers import AutoTokenizer

# # ----------------------------
# # CLI ARGS
# # ----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--dgx", default="http://127.0.0.1:8000")
# parser.add_argument("--draft_n", type=int, default=4)
# parser.add_argument("--max_tokens", type=int, default=80)
# parser.add_argument("--prompt", type=str, default="Count from 1 to 10 goes like")
# args = parser.parse_args()

# DGX_URL = args.dgx.rstrip("/")
# DRAFT_N = args.draft_n
# MAX_TOKENS = args.max_tokens
# PROMPT = args.prompt

# # ----------------------------
# # MODEL CONFIG (Quantized)
# # ----------------------------
# # MODEL_REPO = "TheBloke/Mistral-1.1B-Instruct-GGUF"
# # MODEL_FILE = "mistral-1.1b-instruct.Q4_K_M.gguf"
# # TOKENIZER_REPO = "mistralai/Mistral-1.1B-Instruct"

# MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# model_type = "llama"



# print("\n============================================================")
# print("LOADING DRAFT MODEL (Apple Silicon 4-bit)")
# print("============================================================")
# print(f"Model: {MODEL_REPO}/{MODEL_FILE}")

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_REPO,
#     model_file=MODEL_FILE,
#     model_type="llama",
#     gpu_layers=60,
# )
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
# print("✓ Model and tokenizer loaded successfully")

# # ----------------------------
# # PREFILL PHASE
# # ----------------------------
# print("\n============================================================")
# print("PREFILL PHASE")
# print("============================================================")
# print(f"Prompt: {PROMPT}")

# r = requests.post(f"{DGX_URL}/prefill/", json={"prompt": PROMPT})
# r.raise_for_status()
# prompt_len = r.json().get("prompt_len", 0)
# print(f"✓ DGX prefill ready. Prompt length = {prompt_len} tokens")

# tokens = tokenizer.encode(PROMPT)
# device = "mps"

# # Aligned generation params
# GEN_PARAMS = dict(
#     n_predict=DRAFT_N,
#     temp=0.8,              # matches verifier sampling temperature
#     top_k=40,
#     top_p=0.9,
#     repeat_penalty=1.1,
# )

# print("\n============================================================")
# print("SPECULATIVE DECODING LOOP")
# print("============================================================")
# print(f"Target: {MAX_TOKENS} tokens | Draft window: {DRAFT_N}\n")

# start_time = time.time()
# stats = {"draft_time":0.0, "verify_time":0.0, "accepted":0, "drafted":0, "iters":0}

# while len(tokens) < MAX_TOKENS:
#     iter_start = time.time()
#     print(f"\n--- Iteration {stats['iters']+1} --- ({len(tokens)}/{MAX_TOKENS})")

#     # Draft tokens
#     t0 = time.time()
#     out = model.generate(tokens, **GEN_PARAMS)
#     draft_tokens = out[len(tokens):]
#     stats["draft_time"] += time.time() - t0

#     if not draft_tokens:
#         print("⚠️  No draft tokens.")
#         break

#     stats["drafted"] += len(draft_tokens)
#     draft_text = tokenizer.decode(draft_tokens, skip_special_tokens=True)
#     print(f"📝 Drafted {len(draft_tokens)} tokens: {repr(draft_text)[:60]}")

#     # Verify on DGX
#     v0 = time.time()
#     r = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": draft_tokens})
#     verify_resp = r.json()
#     stats["verify_time"] += time.time() - v0

#     accepted = verify_resp.get("accepted_prefix_len", 0)
#     if accepted > 0:
#         accepted_text = tokenizer.decode(draft_tokens[:accepted], skip_special_tokens=True)
#         tokens.extend(draft_tokens[:accepted])
#         stats["accepted"] += accepted
#         print(f"✅ Accepted {accepted}/{len(draft_tokens)} → {repr(accepted_text)[:60]}")
#     else:
#         print(f"❌ Rejected all {len(draft_tokens)} tokens.")
#         break

#     stats["iters"] += 1
#     if len(tokens) >= MAX_TOKENS: break

# # ----------------------------
# # SUMMARY
# # ----------------------------
# total_time = time.time() - start_time
# gen_text = tokenizer.decode(tokens, skip_special_tokens=True)

# print("\n============================================================")
# print("GENERATION COMPLETE")
# print("============================================================")
# print(f"Tokens accepted: {stats['accepted']}/{stats['drafted']} ({100*stats['accepted']/max(1,stats['drafted']):.1f}%)")
# print(f"Total time: {total_time:.2f}s, Tokens/sec: {len(tokens)/max(1,total_time):.2f}")
# print(f"Final text:\n{'='*60}\n{gen_text}\n{'='*60}")


# # # draft_token_generator.py
# # # Fixed version with proper greedy decoding and attention mask

# # import argparse
# # import time
# # import requests
# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # parser = argparse.ArgumentParser()
# # parser.add_argument("--dgx", default="http://127.0.0.1:8000")
# # # parser.add_argument("--model", default="distilgpt2")
# # parser.add_argument("--model", default="gpt2")

# # parser.add_argument("--draft_n", type=int, default=4)
# # parser.add_argument("--max_tokens", type=int, default=100)
# # parser.add_argument("--prompt", type=str, default="Explain quantum entanglement simply.")
# # args = parser.parse_args()

# # DGX_URL = args.dgx.rstrip("/")
# # DRAFT_N = args.draft_n
# # MAX_TOKENS = args.max_tokens
# # PROMPT = args.prompt

# # print(f"Loading drafter model {args.model} ...")
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
# # tokenizer = AutoTokenizer.from_pretrained(args.model)

# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token

# # stats = {
# #     "total_drafted": 0,
# #     "total_accepted": 0,
# #     "iterations": 0,
# #     "draft_time": 0.0,
# #     "verify_time": 0.0,
# #     "acceptance_by_iter": [],
# # }

# # print(f"\n{'='*60}")
# # print(f"PREFILL PHASE")
# # print(f"{'='*60}")
# # print(f"Prompt: {PROMPT}")

# # r = requests.post(f"{DGX_URL}/prefill/", json={"prompt": PROMPT})
# # if r.status_code != 200:
# #     print(f"❌ Prefill failed: {r.text}")
# #     raise SystemExit(1)

# # prefill_resp = r.json()
# # prompt_len = prefill_resp.get("prompt_len", 0)
# # print(f"✓ DGX prefill ready. Prompt length = {prompt_len} tokens")

# # tokens = tokenizer.encode(PROMPT)
# # committed = len(tokens)

# # print(f"\n{'='*60}")
# # print(f"SPECULATIVE DECODING LOOP")
# # print(f"{'='*60}")
# # print(f"Target: {MAX_TOKENS} tokens | Draft window: {DRAFT_N}\n")

# # start_time = time.time()
# # iteration = 0

# # while len(tokens) < MAX_TOKENS:
# #     iteration += 1
# #     iter_start = time.time()
    
# #     print(f"\n--- Iteration {iteration} ---")
# #     print(f"Current: {len(tokens)}/{MAX_TOKENS} tokens")
    
# #     # Generate draft tokens - GREEDY with proper attention mask
# #     draft_start = time.time()
# #     input_ids = torch.tensor([tokens]).to(device)
# #     attention_mask = torch.ones_like(input_ids)  # Proper attention mask
    
# #     with torch.no_grad():
# #         out = model.generate(
# #             input_ids,
# #             attention_mask=attention_mask,  # Add attention mask
# #             max_new_tokens=DRAFT_N,
  
# #             do_sample=True,  # From False to True
# #             top_k=50,
# #             pad_token_id=tokenizer.eos_token_id,
# #         )
    
# #     generated = out[0].cpu().tolist()[len(tokens):]
# #     draft_time = time.time() - draft_start
# #     stats["draft_time"] += draft_time
    
# #     if len(generated) == 0:
# #         print("⚠️  No more tokens. Stopping.")
# #         break
    
# #     stats["total_drafted"] += len(generated)
# #     draft_text = tokenizer.decode(generated, skip_special_tokens=True)
# #     print(f"📝 Drafted {len(generated)} tokens: {repr(draft_text)[:60]}")
# #     print(f"   Token IDs: {generated}")
    
# #     # Verify with DGX
# #     verify_start = time.time()
# #     r = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": generated})
# #     verify_time = time.time() - verify_start
# #     stats["verify_time"] += verify_time
    
# #     if r.status_code != 200:
# #         print(f"❌ Verify failed: {r.text}")
# #         break
    
# #     resp = r.json()
# #     accepted = resp.get("accepted_prefix_len", 0)
# #     preds = resp.get("preds", [])
    
# #     if accepted > 0:
# #         accepted_text = tokenizer.decode(generated[:accepted], skip_special_tokens=True)
# #         tokens.extend(generated[:accepted])
# #         stats["total_accepted"] += accepted
# #         stats["acceptance_by_iter"].append(accepted)
        
# #         acceptance_rate = (accepted / len(generated)) * 100
# #         print(f"✅ Accepted {accepted}/{len(generated)} ({acceptance_rate:.0f}%): {repr(accepted_text)[:60]}")
# #         print(f"   Output so far: ...{repr(tokenizer.decode(tokens[-20:]))}")
# #         committed = len(tokens)
# #     else:
# #         print(f"❌ Rejected all {len(generated)} tokens")
# #         stats["acceptance_by_iter"].append(0)
        
# #         # Show why rejected
# #         if len(preds) > 0:
# #             print(f"   Expected: {preds[:3]}... Got: {generated[:3]}...")
        
# #         # Fallback: single token
# #         print("   Trying fallback (1 token)...")
# #         with torch.no_grad():
# #             out2 = model.generate(
# #                 input_ids,
# #                 attention_mask=attention_mask,
# #                 max_new_tokens=1,
# #                 do_sample=False,
# #                 pad_token_id=tokenizer.eos_token_id,
# #             )
        
# #         gen1 = out2[0].cpu().tolist()[len(tokens):]
# #         if not gen1:
# #             print("⚠️  Cannot generate. Stopping.")
# #             break
        
# #         stats["total_drafted"] += 1
# #         r2 = requests.post(f"{DGX_URL}/verify/", json={"draft_tokens": gen1})
# #         resp2 = r2.json()
# #         accepted2 = resp2.get("accepted_prefix_len", 0)
        
# #         if accepted2 > 0:
# #             tokens.extend(gen1)
# #             stats["total_accepted"] += 1
# #             fallback_text = tokenizer.decode(gen1, skip_special_tokens=True)
# #             print(f"   ✓ Fallback accepted: {repr(fallback_text)}")
# #             committed = len(tokens)
# #         else:
# #             print("   ❌ Fallback rejected. Aborting.")
# #             break
    
# #     stats["iterations"] += 1
# #     iter_time = time.time() - iter_start
# #     print(f"⏱️  {iter_time:.2f}s (draft: {draft_time:.2f}s, verify: {verify_time:.2f}s)")

# # end_time = time.time()
# # total_time = end_time - start_time
# # new_tokens = len(tokens) - prompt_len

# # # Statistics
# # print(f"\n{'='*60}")
# # print(f"GENERATION COMPLETE")
# # print(f"{'='*60}")
# # print(f"\n📊 Token Statistics:")
# # print(f"   Prompt tokens:    {prompt_len}")
# # print(f"   Generated tokens: {new_tokens}")
# # print(f"   Total tokens:     {len(tokens)}")
# # print(f"\n⚡ Performance:")
# # print(f"   Total time:       {total_time:.2f}s")
# # if new_tokens > 0:
# #     print(f"   Tokens/second:    {new_tokens/total_time:.1f}")
# # print(f"   Draft time:       {stats['draft_time']:.2f}s ({stats['draft_time']/total_time*100:.0f}%)")
# # print(f"   Verify time:      {stats['verify_time']:.2f}s ({stats['verify_time']/total_time*100:.0f}%)")

# # if stats['total_drafted'] > 0:
# #     print(f"\n🎯 Speculative Decoding:")
# #     print(f"   Iterations:       {stats['iterations']}")
# #     print(f"   Tokens drafted:   {stats['total_drafted']}")
# #     print(f"   Tokens accepted:  {stats['total_accepted']}")
# #     acceptance_rate = (stats['total_accepted'] / stats['total_drafted']) * 100
# #     print(f"   Acceptance rate:  {acceptance_rate:.1f}%")
    
# #     if new_tokens > 0 and stats['iterations'] > 0:
# #         avg_verify_time = stats['verify_time'] / stats['iterations']
# #         estimated_regular = new_tokens * avg_verify_time
# #         speedup = estimated_regular / total_time if total_time > 0 else 1.0
# #         print(f"\n🚀 Speedup Estimate:")
# #         print(f"   Regular: {estimated_regular:.2f}s vs Speculative: {total_time:.2f}s")
# #         print(f"   Speedup: {speedup:.2f}x")

# # print(f"\n📝 Final Output:")
# # print(f"{'='*60}")
# # output_text = tokenizer.decode(tokens)
# # print(output_text)
# # print(f"{'='*60}")

# # if stats['acceptance_by_iter']:
# #     print(f"\n📈 Acceptance per iteration: {stats['acceptance_by_iter']}")
