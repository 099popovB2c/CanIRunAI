# CanIRunAI

CanIRunAI inspects your computer and estimates which local AI models are realistic **before** you download them.

## v0.4.0

- Local benchmark calibration: record real token-per-second results from your own machine
- Calibrated speed ranges use nearby hardware samples instead of only generic heuristics
- Benchmark history stays in a local JSON file
- `--compare` filters the catalog to named models/tags for side-by-side checks
- JSON reports include calibration factor and whether each speed range is heuristic or calibrated
- Existing context/KV-cache memory model, max-context estimate and best-fit recommendations retained

```bash
python canirunai.py
python canirunai.py --compare "Qwen3,Gemma"
python canirunai.py --record-benchmark "Qwen3 8B" 18.7
python canirunai.py --list-benchmarks
python canirunai.py --benchmark-file my-benchmarks.json --json report.json
```

Record the tok/s value reported by the runtime you actually use. Calibration is approximate because prompt length, backend, GPU offload, batch size and model architecture still matter.
