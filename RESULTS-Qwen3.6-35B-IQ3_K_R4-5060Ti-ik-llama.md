# Qwen3.6-35B-A3B IQ3_K_R4 on RTX 5060 Ti 16GB — ik_llama.cpp

**TL;DR**: 128–129 t/s text generation, essentially flat from 0 to 139k context, on a $429 consumer GPU using ik_llama.cpp's R4 quant format. Faster than the RTX 5070 Ti on mainline llama.cpp at every context depth.

## Hardware

- **GPU**: RTX 5060 Ti 16GB (Blackwell sm_120, GDDR7 ~448 GB/s), 15,825 MiB reported by CUDA
- **CPU/RAM**: Ryzen 9 7900X, 64 GB DDR5
- **OS**: Ubuntu 24.04 desktop (display server active, ~1 GB VRAM in use before inference)
- **Inference**: ik_llama.cpp (commit b8eb8cc), CUDA 12.8, sm_89;120

## Model

- **Source GGUF**: `unsloth/Qwen3.6-35B-A3B-GGUF` → `Qwen3.6-35B-A3B-Q8_0.gguf` (36.9 GB)
- **Quant**: `IQ3_K_R4` — ik_llama.cpp-specific format, reorders data for CPU+GPU split
- **Output file**: `/data/models/qwen3.6-35b-a3b-instruct-pure/Qwen3.6-35B-A3B-IQ3_K_R4.gguf`
- **Size**: 15 GB on disk, **14.28 GiB as loaded by ggml** (3.4325 bpw)
- **Config**: `--n-cpu-moe 3` → 3/41 expert layers on CPU, 38/41 on GPU

## Build — ik_llama.cpp

System cmake (3.22.1) is too old; requires cmake ≥3.25 for CUDA20. Use pip cmake.

```bash
# Install cmake via pip (system cmake 3.22 too old for CUDA20 support)
/home/abovespec/friendnet-mvp/.venv/bin/pip install cmake
NEW_CMAKE=/home/abovespec/friendnet-mvp/.venv/lib/python3.11/site-packages/cmake/data/bin/cmake

# Clone
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp /data/ik-llama

# Configure
$NEW_CMAKE -S /data/ik-llama -B /data/ik-llama/build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DGGML_NATIVE=ON \
  -DGGML_CCACHE=ON

# Build (10–15 min)
make -C /data/ik-llama/build llama-quantize llama-bench llama-server -j$(nproc)
```

Binaries at: `/data/ik-llama/build/bin/`

## Model Download & Quantization

```bash
# Download Q8_0 source (36.9 GB) — /data has 642 GB free
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "*Q8_0*" \
  --local-dir /data/models/qwen3.6-35b-a3b-instruct-pure/

# Quantize to IQ3_K_R4 (~2 min)
/data/ik-llama/build/bin/llama-quantize \
  --allow-requantize \
  --leave-output-tensor \
  /data/models/qwen3.6-35b-a3b-instruct-pure/Qwen3.6-35B-A3B-Q8_0.gguf \
  /data/models/qwen3.6-35b-a3b-instruct-pure/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
  IQ3_K_R4
```

**Note**: `--allow-requantize` is needed because Q8_0 tensors are blocked by default.
`--leave-output-tensor` preserves the output layer at higher precision for quality.
Starting from Q5_K_M fails (has q6_K tensors which are also blocked).
Ideal source would be BF16 (69.4 GB) but Q8_0 is sufficient for speed testing.

## Run — daily driver server

```bash
/data/ik-llama/build/bin/llama-server \
  --model /data/models/qwen3.6-35b-a3b-instruct-pure/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
  --alias qwen3.6-35b-iq3-r4 \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --n-gpu-layers 99 \
  --n-cpu-moe 3 \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --threads 12 \
  --batch-size 2048 --ubatch-size 512
```

## Benchmark Commands

```bash
# Full depth sweep (q4_0 KV) — TG at each depth
for depth in 0 16384 32768 65536 98304 131072 139264; do
  echo -n "depth=${depth}: "
  /data/ik-llama/build/bin/llama-bench \
    -m /data/models/qwen3.6-35b-a3b-instruct-pure/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
    -ngl 99 --n-cpu-moe 3 -fa 1 \
    -ctk q4_0 -ctv q4_0 \
    -p ${depth} -n 128 -r 2 2>/dev/null \
    | grep "tg128" | grep -oP '[\d.]+(?= ±)' | head -1
done

# PP sweep (q4_0 KV) — all depths in one run
/data/ik-llama/build/bin/llama-bench \
  -m /data/models/qwen3.6-35b-a3b-instruct-pure/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
  -ngl 99 --n-cpu-moe 3 -fa 1 \
  -ctk q4_0 -ctv q4_0 \
  -p 0,16384,32768,65536,98304,131072,139264 -n 128 -r 1
```

## Results

### TG Speed at Depth — q4_0 KV, ncmoe=3

| ctx depth | tg128 t/s |
|---:|---:|
| 0 | 128.58 |
| 16k | 128.78 |
| 32k | 128.91 |
| 65k | 128.35 |
| 98k | 128.01 |
| 131k | 127.85 |
| **139k** | **126.10** |

**Only 2% drop from empty to max context.** The expert FFN computation (38/41 layers on GPU) dominates generation time and is context-independent. Flash-attention overhead is negligible relative to expert compute at this ncmoe value.

### Prompt Processing Speed — q4_0 KV, ncmoe=3

| ctx depth | pp512 t/s |
|---:|---:|
| 0 | ~1754 |
| 16k | 1814 |
| 32k | 1762 |
| 65k | 1601 |
| 98k | 1470 |
| 131k | 1358 |
| 139k | 1292 |

### Max Context Ceiling

| KV type | max ctx (loads) | next that OOMs |
|---|---:|---:|
| q8_0 | ~84k (83,968) | ~86k (86,016) |
| **q4_0** | **~141k (141,312)** | **~143k (143,360)** |

### Speed at Empty Context — KV type comparison

| KV type | tg128 t/s | max ctx |
|---|---:|---:|
| q8_0 | ~122 | ~84k |
| q4_0 | ~129 | ~141k |

q4_0 is faster AND gives more context — recommended for all use cases.

## Comparison: 5060 Ti ik_llama.cpp vs 5070 Ti mainline llama.cpp

| GPU | fork | quant | ncmoe | ctx depth | tg128 |
|---|---|---|---|---|---:|
| RTX 5070 Ti 16GB | mainline | Q4_K_S | 15 | 0 | 100.4 |
| RTX 5070 Ti 16GB | mainline | Q4_K_S | 15 | 65k | 69.2 |
| RTX 5070 Ti 16GB | mainline | Q4_K_S | 15 | 131k | ~55 (est.) |
| **RTX 5060 Ti 16GB** | **ik_llama.cpp** | **IQ3_K_R4** | **3** | **0** | **128.6** |
| **RTX 5060 Ti 16GB** | **ik_llama.cpp** | **IQ3_K_R4** | **3** | **65k** | **128.4** |
| **RTX 5060 Ti 16GB** | **ik_llama.cpp** | **IQ3_K_R4** | **3** | **131k** | **127.9** |

The 5060 Ti with ik_llama.cpp and IQ3_K_R4 outperforms the 5070 Ti on mainline at every context depth, despite being a cheaper card with a more aggressive quantization.

## Context on IQ3_K_R4 Quantization Quality

- Source was Q8_0 re-quantized with `--allow-requantize` (not ideal — from full precision BF16 would be best)
- Quality is lower than IQ4_K_R4 or the Q4_K_S used on the 5070 Ti
- For a proper quality comparison, IQ4_K_R4 from BF16 (69.4 GB) would be the right quant
- Speed numbers are valid regardless of the quantization quality

## TODO / Next Tests

- [ ] Test IQ4_K_R4 from BF16 for better quality (69.4 GB download required)
- [ ] Compare quality: IQ3_K_R4 vs Q4_K_S via perplexity
- [ ] Test higher ncmoe values to see context/speed tradeoff
- [ ] Compare with mainline llama.cpp IQ3_XS (full VRAM) as proper baseline: ~60 t/s at 16k

## Key Flags Reference

| flag | meaning |
|---|---|
| `--n-cpu-moe 3` | keep 3 of 41 expert layers on CPU, rest on GPU |
| `-ctk q4_0 -ctv q4_0` | 4-bit quantized KV cache |
| `-ctk q8_0 -ctv q8_0` | 8-bit quantized KV cache (less ctx, slightly slower) |
| `-fa 1` | FlashAttention on |
| `-ngl 99` | all layers to GPU (except those governed by --n-cpu-moe) |
| `IQ3_K_R4` | ik_llama.cpp R4 quant: reordered for CPU+GPU split, ~3.4 bpw |
