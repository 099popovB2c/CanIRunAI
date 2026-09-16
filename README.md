# CanIRunAI

CanIRunAI inspects your computer and estimates which local AI model sizes are realistic **before** you download them.

## v0.2.0

- Detect total/available RAM, disk, CPU and all NVIDIA GPUs
- Aggregate multi-GPU VRAM
- Detect Ollama, llama.cpp, LM Studio CLI and vLLM when present
- Quantization-aware estimates: Q2_K through Q8_0
- Conservative GOOD / TIGHT / SLOW / NO fit verdicts
- Hardware overrides to simulate another machine
- Evaluate an arbitrary model size with `--model-size`
- JSON output

```bash
python canirunai.py
python canirunai.py --quant Q5_K_M
python canirunai.py --model-size 12 --vram 12 --ram 32
python canirunai.py --json report.json
```

Estimates are approximate: context length, KV cache, architecture, runtime and offload settings materially affect real memory and speed.
