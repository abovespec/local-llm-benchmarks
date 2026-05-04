# Replication Guide — Qwen3.6-27B IQ3_K_R4 + Memory OC on RTX 5060 Ti

**Result**: ~28 t/s stock → ~33.8 t/s at +5000 MHz memory OC, flat from 0 → 139k context, on an RTX 5060 Ti 16 GB using ik_llama.cpp.

The OC is free performance — no voltage change, no core OC. Just memory bandwidth.

Tested on: RTX 5060 Ti 16 GB (Blackwell, driver 580.126.20). The OC approach may work on other RTX 50-series cards — silicon lottery applies, your ceiling may differ.

---

## What you need

- RTX 5060 Ti 16 GB (or similar 16 GB Blackwell GPU)
- Ubuntu 22.04 or 24.04
- ~40 GB free disk space (27 GB for Q8_0 source + 12 GB for IQ3_K_R4 output)
- CUDA 12.x installed
- Python 3.10+ with pip (for cmake workaround)
- Rust toolchain (for nvoc — the OC tool)

CPU RAM is not a bottleneck. The model is fully GPU-resident.

---

## Step 1 — Fix cmake (Ubuntu only)

Ubuntu's system cmake is too old for CUDA 20 support. Install a newer one via pip.

```bash
pip install cmake --user

# Find the path — you'll need it in Step 2
python3 -c "import cmake; import os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin', 'cmake'))"
```

---

## Step 2 — Build ik_llama.cpp

```bash
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp /data/ik-llama
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU:

| GPU generation | CUDA arch |
|---|---|
| RTX 3000 series (Ampere) | `86` |
| RTX 4000 series (Ada Lovelace) | `89` |
| RTX 5000 series (Blackwell) | `120` |

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

---

## Step 3 — Get the model

**Option A — Download the pre-built IQ3_K_R4 directly (easiest, ~12 GB):**

```bash
hf download abovespec/Qwen3.6-27B-IQ3_K_R4-GGUF \
  "Qwen3.6-27B-IQ3_K_R4.gguf" \
  --local-dir /data/models/qwen3.6-27b/
```

HuggingFace repo: https://huggingface.co/abovespec/Qwen3.6-27B-IQ3_K_R4-GGUF

**Option B — Quantize from Q8_0 yourself (~27 GB source):**

```bash
# Download Q8_0 source
hf download unsloth/Qwen3.6-27B-GGUF \
  "Qwen3.6-27B-Q8_0.gguf" \
  --local-dir /data/models/qwen3.6-27b/

# Quantize to IQ3_K_R4 (~2 minutes)
/data/ik-llama/build/bin/llama-quantize \
  --allow-requantize \
  --leave-output-tensor \
  /data/models/qwen3.6-27b/Qwen3.6-27B-Q8_0.gguf \
  /data/models/qwen3.6-27b/Qwen3.6-27B-IQ3_K_R4.gguf \
  IQ3_K_R4
```

---

## Step 4 — Install nvoc (memory overclock tool)

Standard OC tools don't work on Blackwell. nvoc uses the correct NVML call for RTX 50-series.

```bash
# Requires Rust — install if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/martinstark/nvoc
cd nvoc && cargo build --release
sudo cp target/release/nvoc /usr/local/bin/
```

Verify:
```bash
nvoc info
```

---

## Step 5 — Apply the OC

```bash
sudo nvoc -m 5000   # +5000 MHz memory offset
nvoc info           # confirm mem offset shows +5000MHz
```

**Silicon lottery**: +5000 MHz was the stable ceiling on this specific card. Start lower (+1000, +2000) and step up. If the system crashes, drop back by 500 MHz. +6000 MHz crashed immediately on this card.

The offset resets on reboot. To make it persistent, create a systemd service:

```bash
sudo tee /etc/systemd/system/nvoc.service > /dev/null << 'EOF'
[Unit]
Description=Apply GPU memory overclock
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/nvoc -m 5000
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now nvoc.service
```

---

## Step 6 — Benchmark

```bash
/data/ik-llama/build/bin/llama-bench \
  -m /data/models/qwen3.6-27b/Qwen3.6-27B-IQ3_K_R4.gguf \
  -ngl 99 -fa 1 \
  -ctk q4_0 -ctv q4_0 \
  -p 0,16384,65536,131072 -n 128 -r 3
```

Expected results at +5000 MHz OC (RTX 5060 Ti):

| Context depth | TG t/s |
|---:|---:|
| 0 | ~32.9 |
| 16k | ~33.9 |
| 65k | ~33.8 |
| 131k | ~33.9 |

Stock (no OC): ~28.3 t/s at all depths.

---

## Step 7 — Run as a server

```bash
/data/ik-llama/build/bin/llama-server \
  --model /data/models/qwen3.6-27b/Qwen3.6-27B-IQ3_K_R4.gguf \
  -ngl 99 \
  -fa 1 \
  -ctk q4_0 -ctv q4_0 \
  -c 131072 \
  --temp 0.6 \
  --jinja \
  --port 8080
```

Note: unlike the 35B MoE model, the 27B dense model does **not** need `--n-cpu-moe`. All layers fit on GPU entirely.

### Key flags

| Flag | What it does |
|---|---|
| `-ngl 99` | All layers on GPU |
| `-fa 1` | FlashAttention |
| `-ctk q4_0 -ctv q4_0` | Quantized KV cache — more context, less VRAM |
| `-c 131072` | 131k context (safe max; 139k also works, OOM at 147k) |
| `--jinja` | Jinja chat templates (required for Qwen3.6 thinking mode + tool calling) |

---

## VRAM budget

| Component | VRAM |
|---|---|
| Model (IQ3_K_R4) | 11.53 GiB |
| KV cache at 131k (q4_0) | ~0.9 GiB |
| Compute buffer | ~0.4 GiB |
| **Total** | **~12.8 GiB** |

Fits comfortably in 16 GB. Unlike the 35B MoE model, no CPU expert offloading needed.

---

## Troubleshooting

**OOM at launch:** Reduce context with `-c 65536`. Or check that no other process is holding VRAM.

**"IQ3_K_R4 not supported":** You're running mainline llama.cpp, not ik_llama.cpp. Use the binary from `/data/ik-llama/build/bin/`.

**Speed below ~27 t/s (stock):** Check `nvidia-smi` — confirm `-ngl 99` is working and the model is on GPU.

**OC crash:** The system will reboot or freeze. This is expected at too-high offsets — no permanent damage. Drop to a lower value.

**nvoc offset not applying:** Make sure you're running with `sudo`. Check `nvoc info` after applying.
