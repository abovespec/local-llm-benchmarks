# Replication Guide — Qwen3.6-35B at 128 t/s on a 16 GB NVIDIA GPU

**Result**: 128–129 t/s text generation, flat from 0 → 139k context, on an RTX 5060 Ti 16 GB using ik_llama.cpp's IQ3_K_R4 quant format. Beats the RTX 5070 Ti on mainline llama.cpp at every context depth.

Tested on: RTX 5060 Ti 16 GB (Blackwell). Should work on any 16 GB NVIDIA GPU with sm_86+ (RTX 3090, 4060 Ti 16 GB, 4080, 4090, 5060 Ti, 5070, 5080, 5090 — adjust CUDA arch below).

---

## What you need

- 16 GB NVIDIA GPU (sm_86+)
- Ubuntu 22.04 or 24.04 (other Linux distros should work with minor adjustments)
- ~60 GB free disk space (36.9 GB for Q8_0 source + 15 GB for IQ3_K_R4 output)
- CUDA 12.x installed
- Python 3.10+ with pip (for cmake workaround — see below)

CPU RAM is not a bottleneck. The model lives almost entirely in VRAM. Even 32 GB RAM is fine.

---

## Step 1 — Fix cmake (Ubuntu only)

Ubuntu's system cmake is too old (3.22) for CUDA 20 support. Install a newer one via pip.

```bash
# Install into any venv or user pip — just needs to be accessible
pip install cmake --user

# Find where it landed
python3 -c "import cmake; import os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin', 'cmake'))"
```

Note the path — you'll use it in Step 2. It will look something like:
`~/.local/lib/python3.11/site-packages/cmake/data/bin/cmake`

---

## Step 2 — Build ik_llama.cpp

```bash
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp /data/ik-llama
```

Configure — replace `NEW_CMAKE` with your path from Step 1, and adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU:

| GPU generation | CUDA arch |
|---|---|
| RTX 3000 series (Ampere) | `86` |
| RTX 4000 series (Ada Lovelace) | `89` |
| RTX 5000 series (Blackwell) | `120` |
| Mixed / safe default | `86;89;120` |

```bash
NEW_CMAKE=~/.local/lib/python3.11/site-packages/cmake/data/bin/cmake  # adjust to your path

$NEW_CMAKE -S /data/ik-llama -B /data/ik-llama/build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DGGML_NATIVE=ON \
  -DGGML_CCACHE=ON

make -C /data/ik-llama/build llama-quantize llama-bench llama-server -j$(nproc)
```

Build takes 10–15 minutes. Binaries land in `/data/ik-llama/build/bin/`.

> **Tip**: If you hit cmake/CUDA build errors, Claude Code is useful for debugging — just paste the error and it'll walk you through it.

---

## Step 3 — Download the source model

IQ3_K_R4 is an ik_llama.cpp-specific format — no pre-built version exists. You build it yourself from a Q8_0 source.

```bash
# Install hf CLI if you don't have it
pip install huggingface-hub --user

# Download Q8_0 source — 36.9 GB
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  "Qwen3.6-35B-A3B-Q8_0.gguf" \
  --local-dir /data/models/qwen3.6-35b/
```

---

## Step 4 — Quantize to IQ3_K_R4

Takes about 2 minutes. Output is 15 GB.

```bash
/data/ik-llama/build/bin/llama-quantize \
  --allow-requantize \
  --leave-output-tensor \
  /data/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q8_0.gguf \
  /data/models/qwen3.6-35b/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
  IQ3_K_R4
```

- `--allow-requantize` — required because Q8_0 tensors are blocked by default
- `--leave-output-tensor` — keeps the final output layer at higher precision (better quality)

> **Note**: Starting from Q5_K_M will fail — it contains q6_K tensors which are also blocked. Use Q8_0 or BF16 as your source.

---

## Step 5 — Launch the server

```bash
/data/ik-llama/build/bin/llama-server \
  --model /data/models/qwen3.6-35b/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
  -ngl 99 \
  --n-cpu-moe 3 \
  -fa 1 \
  -ctk q4_0 -ctv q4_0 \
  -c 131072 \
  --temp 0.2 \
  --jinja \
  --port 8080
```

Verify it's up:
```bash
curl http://localhost:8080/health
```

### Key flags explained

| Flag | What it does |
|---|---|
| `-ngl 99` | Load all layers onto GPU |
| `--n-cpu-moe 3` | Keep 3 of 41 MoE expert layers in CPU RAM, freeing ~1.5 GB VRAM for the compute buffer |
| `-fa 1` | Enable FlashAttention |
| `-ctk q4_0 -ctv q4_0` | 4-bit quantized KV cache — faster AND gives more context than q8_0 |
| `-c 131072` | 131k context window (safe max at q4_0 KV) |
| `--temp 0.2` | Sensible default temperature |
| `--jinja` | Use Jinja chat templates (required for Qwen3.6's thinking mode) |

---

## Step 6 — Verify your speed (optional)

Run the benchmark to confirm you're hitting expected numbers:

```bash
/data/ik-llama/build/bin/llama-bench \
  -m /data/models/qwen3.6-35b/Qwen3.6-35B-A3B-IQ3_K_R4.gguf \
  -ngl 99 --n-cpu-moe 3 -fa 1 \
  -ctk q4_0 -ctv q4_0 \
  -p 0,16384,65536,131072 -n 128 -r 2
```

Expected TG results (RTX 5060 Ti):

| Context depth | t/s |
|---:|---:|
| 0 | ~128.6 |
| 16k | ~128.8 |
| 65k | ~128.4 |
| 131k | ~127.9 |

The curve is essentially flat. Only 2% drop from empty to max context.

---

## VRAM budget breakdown

With `--n-cpu-moe 3` on the 5060 Ti 16 GB:

| Component | VRAM |
|---|---|
| Model (IQ3_K_R4) | 14.28 GiB |
| KV cache at 131k (q4_0) | ~1.0 GiB |
| Compute buffer | ~0.5 GiB |
| **Total** | **~15.8 GiB** |

If you get OOM, increase `--n-cpu-moe` to 4 or 5. Each extra expert layer offloaded to CPU frees roughly 400–500 MB of VRAM with a small speed penalty.

---

## Troubleshooting

**OOM at launch:**
Increase `--n-cpu-moe` (try 4, 5, 6). Or reduce context: `-c 65536`.

**cmake too old error:**
Follow Step 1. The pip cmake workaround is reliable.

**"IQ3_K_R4 not supported" error on server:**
You're using mainline llama.cpp, not ik_llama.cpp. Make sure you're running the binary from `/data/ik-llama/build/bin/`.

**Slow speed (far below 128 t/s):**
Check `nvidia-smi` — confirm the model is on GPU, not falling back to CPU. Make sure `-ngl 99` is set.

**Q5_K_M requantize fails:**
Use Q8_0 or BF16 as source. Q5_K_M contains q6_K attention tensors that are blocked.

---

## Notes on quantization quality

IQ3_K_R4 at 3.44 bpw is an aggressive quant. Quality is lower than Q4_K_M but perfectly usable for coding tasks, tool-calling, and creative generation — as demonstrated in the accompanying thread.

For better quality at the cost of ~4 GB more VRAM, build IQ4_K_R4 instead (needs partial CPU offload on 16 GB):

```bash
/data/ik-llama/build/bin/llama-quantize \
  --allow-requantize \
  --leave-output-tensor \
  /data/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q8_0.gguf \
  /data/models/qwen3.6-35b/Qwen3.6-35B-A3B-IQ4_K_R4.gguf \
  IQ4_K_R4
```

Launch with `--n-cpu-moe 12` or higher to fit in 16 GB. Speed will drop to ~60–80 t/s but quality improves noticeably.

---

## AMD GPU?

ik_llama.cpp can be built with ROCm (`-DGGML_HIPBLAS=ON` instead of `-DGGML_CUDA=ON`). The base inference works. However, the IQ3_K_R4 / IQ4_K_R4 R4 format uses custom CUDA kernels that don't have HIP equivalents yet — AMD users can run the fork but won't get the R4 speed benefits. Standard quant types (IQ3_XS, Q4_K_M etc.) run fine on AMD.
