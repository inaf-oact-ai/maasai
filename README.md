<p align="left">
  <img src="share/Logo_MAASAI_noborder.png" alt="MAASAI logo" width="300"/>
</p>

# maasai
Multi-Agent AI System for Astrodata Inference (MAASAI)

This project organizes the MAASAI workflow as a modular LangGraph application.

## Structure

- `maasai/config.py` - environment-backed settings
- `maasai/state.py` - shared graph state
- `maasai/schemas.py` - structured contracts
- `maasai/model_router.py` - LiteLLM alias selector
- `maasai/agents.py` - LangChain agents
- `maasai/guardrails.py` - lightweight early checks
- `maasai/rag.py` - prompt-optimization retrieval stub
- `maasai/tools.py` - astronomy REST/MCP tool stubs
- `maasai/nodes.py` - graph node functions
- `maasai/graph.py` - graph topology
- `main.py` - runnable entry point
- `litellm_config_maasai.yaml` - LiteLLM Proxy config with an upstream pool on port 8000

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run LiteLLM Proxy

```bash
litellm --config litellm_config_maasai.yaml
```

## Run the app

```bash
cp .env.example .env
python main.py
```

## Notes

- The Python app only chooses the logical aliases `astro-small`, `astro-medium`, `astro-large`, and `astro-multimodal`.
- LiteLLM Proxy performs the actual load balancing across the remote vLLM or LLumnix endpoints.
- Replace the placeholder endpoint IPs with your real servers.
- Replace the RAG stub and tool stubs with your production integrations.
