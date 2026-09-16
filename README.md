# CanIRunAI

CanIRunAI answers a simple question before you waste time downloading a model:

> **Can this computer realistically run this local AI model?**

It inspects the local machine, estimates model memory requirements, accounts for quantization and context/KV-cache overhead, and produces a practical fit recommendation.

## What it does

CanIRunAI can help estimate:

- which model sizes fit your hardware
- VRAM/RAM pressure
- effect of Q2 / Q4 / Q5 / Q6 / Q8 quantization
- context-length memory cost
- KV-cache overhead
- likely GPU/RAM offload requirements
- approximate maximum context
- estimated token-per-second range
- whether the estimate is generic or calibrated from your own measurements
- side-by-side comparison between model families

It is intended for people using local runtimes such as Ollama, llama.cpp, LM Studio or similar tools.

## How it works

```text
Detect local hardware
      ↓
RAM / GPU / VRAM / runtime information
      ↓
Select model + quantization + context
      ↓
Estimate model weights + context/KV memory
      ↓
Compare requirement with available hardware
      ↓
Fit recommendation + estimated speed range
```

## Quick start

Run the normal hardware/model analysis:

```bash
python canirunai.py
```

Compare selected model families:

```bash
python canirunai.py --compare "Qwen3,Gemma"
```

Write a JSON report:

```bash
python canirunai.py --json report.json
```

## Quantization

Quantization changes how much memory model weights require. CanIRunAI uses quantization-aware estimates so a model is not treated as if every version has the same footprint.

Typical comparison:

```text
Q4  → smaller memory footprint, usually the practical local default
Q5  → more memory, often somewhat higher quality
Q6  → heavier again
Q8  → much larger memory requirement
```

The exact real-world result varies by model format and runtime implementation, so the figures are estimates rather than byte-perfect guarantees.

## Context and KV cache

Longer context windows consume additional memory. CanIRunAI includes context/KV-cache cost instead of looking only at the model file size.

That matters because a model that fits at a short context may become impractical at a much larger context.

The report therefore considers both:

```text
model weights
+
context / KV-cache overhead
```

## Speed estimates

CanIRunAI can provide an approximate token-per-second range. Generic speed estimates are heuristic because real speed depends on:

- GPU model and memory bandwidth
- CPU
- amount of GPU offload
- backend/runtime
- prompt length
- context size
- batch settings
- model architecture
- quantization format

For that reason v0.4.0 adds **local benchmark calibration**.

## Benchmark calibration

Record a real tok/s value reported by the runtime you actually use:

```bash
python canirunai.py --record-benchmark "Qwen3 8B" 18.7
```

List stored measurements:

```bash
python canirunai.py --list-benchmarks
```

Use a custom benchmark file:

```bash
python canirunai.py --benchmark-file my-benchmarks.json --json report.json
```

CanIRunAI can then adjust future speed estimates using measurements from your machine instead of relying only on generic heuristics.

Benchmark data stays local.

## Model comparison

The comparison mode lets you narrow the catalog to model names/tags and compare practical fit rather than manually calculating each model.

Example:

```bash
python canirunai.py --compare "Qwen3,Gemma"
```

A useful report can answer questions such as:

```text
Which model fits fully in VRAM?
Which one needs RAM offload?
Which quantization is realistic?
How much context can I afford?
Which option is likely to be faster on this machine?
```

## Runtime detection

CanIRunAI can identify common local-AI runtime availability where supported. Runtime detection is informational; it does not silently download or install models.

## Privacy

CanIRunAI is local-first:

- hardware inspection happens locally
- benchmark history stays in a local JSON file
- no CanIRunAI account
- no analytics
- no hosted inference service
- no model upload

## Important limitations

CanIRunAI provides **estimates**, not guaranteed benchmark results.

Two machines with similar RAM/VRAM can perform very differently because of memory bandwidth, backend versions, thermal limits, drivers and offload configuration.

Also:

- model metadata can change between releases
- MoE architectures do not behave exactly like dense models
- runtime-specific memory allocation can differ
- context/KV implementations vary
- token-per-second estimates should be validated with a real benchmark

Use the calibration feature when you want estimates closer to your own machine.

## Roadmap

Possible next steps:

- larger automatically updated model catalog
- richer MoE calculations
- per-runtime memory profiles
- benchmark import from Ollama/llama.cpp logs
- GPU multi-device planning
- one-command recommended Ollama model setup
- community benchmark datasets with explicit opt-in
- desktop GUI

## Version

Current release: **v0.4.0**

## Security

See [SECURITY.md](SECURITY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
