# hetero_specdecode - local prototype

This prototype simulates:
- DGX verifier (server) that does prefill and verification,
- M3 drafter (client) that drafts N tokens and asks verifier to accept them.

## Requirements
Python 3.11.11 and create virtualenvs for each side.

### DGX (verifier)
cd dgx_verifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

### M3 (drafter)
cd Mac_Studio
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python draft_token_generator.py --dgx http://127.0.0.1:8000 --draft_n 4 --max_tokens 80

Notes:
- Default models are `distilgpt2` for verifier and `distilgpt2` for drafter to keep things fast locally.
- To use different models, pass `--model MODEL_NAME` to drafter_client.py (and change MODEL_NAME in verifier_server.py).
- This prototype keeps KV cache in verifier server memory. In production, you will need to stream KV cache from DGX to M3 efficiently.

Optional:
- If you have MLX on Mac and want to use it as the drafter, replace the drafter code to call MLX generate API.

