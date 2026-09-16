# CanIRunAI

CanIRunAI inspects your computer and estimates which local AI models are realistic **before** you download them.

## v0.3.0

- Context-length-aware memory estimates
- KV-cache estimates with FP16, Q8 and Q4 cache modes
- Weight / KV cache / runtime overhead breakdown
- Estimated maximum usable context for detected hardware
- Heuristic token-per-second range, clearly labeled as an estimate rather than a benchmark
- Best-fit model recommendations
- Updated starter catalog with Qwen3, Gemma 3 and gpt-oss families
- Ollama run commands for models that fit
- Existing RAM / VRAM / disk overrides and JSON report output retained

```bash
python canirunai.py
python canirunai.py --context 32768 --kv-quant q8
python canirunai.py --quant Q5_K_M --context 16384
python canirunai.py --model-size 12 --vram 12 --ram 32
python canirunai.py --json report.json
```

Memory and speed numbers are planning estimates. Backend, architecture, prompt length, GPU generation, batch size and offload settings can materially change real results.
