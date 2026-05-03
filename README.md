# local-llm-benchmarks

Benchmarks, replication guides, and results for running large language models locally on consumer GPUs.

All tests run on real hardware — no cloud, no emulation.

---

## Results

| Model | Quant | GPU | t/s TG | Context | Guide |
|---|---|---|---:|---:|---|
| Qwen3.6-35B-A3B | IQ3_K_R4 | RTX 5060 Ti 16 GB | 128–129 | 0→139k (flat) | [guide](REPLICATE-ik-llama-IQ3_K_R4-16GB.md) |

---

## Guides

- [Qwen3.6-35B at 128 t/s on a 16 GB GPU — ik_llama.cpp + IQ3_K_R4](REPLICATE-ik-llama-IQ3_K_R4-16GB.md)

---

## Hardware

- RTX 5060 Ti 16 GB (Blackwell, GDDR7)
- RTX 5070 Ti 16 GB (Blackwell, GDDR7)
- Ryzen 9 7900X, 96 GB DDR5
- Ubuntu 24.04
