# CanIRunAI

CanIRunAI inspects your computer and estimates which local AI model sizes are realistic.

## Features

- Detect RAM, free disk, OS, CPU
- Detect NVIDIA GPU/VRAM via `nvidia-smi` when available
- Model-size profiles from 1.5B to 70B
- Conservative fit recommendations for Q4 inference
- Ollama install command suggestions
- JSON report output

## Run

```bash
python canirunai.py
python canirunai.py --json report.json
```

Performance estimates are approximate and intentionally conservative.
