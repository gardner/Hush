# Hush Lightweight Inference — No PyTorch, No Closed Binaries

## Context

We need to run Hush inference on CPU-only cloud servers without PyTorch in our dependency chain. The current options were:
1. PyTorch + DeepFilterLib (`scripts/infer_single.py`) — heavy, requires PyTorch
2. Closed-source prebuilt binary (`libweya_nc`) — unacceptable for production

## Solution: `scripts/infer_onnx.py`

A single-file inference script using only open-source, auditable dependencies:

- **`onnxruntime`** — runs the three ONNX model files on CPU
- **`DeepFilterLib`** (`libdf`) — lightweight Rust library for DSP (STFT/ISTFT, ERB/DF feature extraction, normalization). Only depends on numpy, no PyTorch.
- **`numpy`** + **`soundfile`** — array ops and WAV I/O

### Verification

Tested against PyTorch reference output (`sample_00006_enhanced.wav`):
- **SNR: 100.9 dB** (essentially bit-perfect, only float32 rounding differences)
- **Correlation: 1.000000**
- **Max absolute difference: 0.000031**

### Usage

```bash
# Install deps (no PyTorch!)
pip install onnxruntime DeepFilterLib numpy soundfile

# Run inference
python scripts/infer_onnx.py \
    --input noisy.wav \
    --output enhanced.wav

# With options
python scripts/infer_onnx.py \
    --model-dir deployment/models/onnx \
    --input noisy.wav \
    --output enhanced.wav \
    --atten-lim-db 20
```

### Architecture

```
Input WAV (16kHz mono)
    ↓
libdf STFT (Vorbis window, FFT=320, hop=160) → [T, 161] complex
    ├→ libdf ERB features → [1, 1, T, 32]
    └→ libdf DF features  → [1, 2, T, 64]
    ↓
ONNX Runtime: enc.onnx(erb, df) → embedding + skip connections
    ├→ ONNX Runtime: erb_dec.onnx → mask [1, 1, T, 32]
    └→ ONNX Runtime: df_dec.onnx  → coefs [1, T, 64, 10]
    ↓
Apply ERB mask (bins 64-160) + DF filter (5-tap FIR, bins 0-63)
    ↓
libdf ISTFT → enhanced WAV
```

## Future: Rust Binary

For a single static binary (no Python at all), port to Rust using:
- `ort` crate — Rust bindings to ONNX Runtime
- Reimplement libdf DSP in Rust (or use the existing Rust `libdf` crate directly)

The Python ONNX script serves as the verified reference implementation.
