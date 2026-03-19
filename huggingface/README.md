---
license: apache-2.0
language: multilingual
tags:
  - speech-enhancement
  - denoising
  - real-time
  - voice-ai
  - hush
  - background-speaker-suppression
  - onnx
  - multilingual
  - audio
  - noise-cancellation
library_name: hush
pipeline_tag: audio-to-audio
---

# Hush

**The first open-source speech enhancement model built specifically for Voice AI — with real-time background speaker suppression.**

> **8 MB model · Runs fully on CPU in real time · Trained on 10,000+ hours of mixed audio · Under 1 ms processing per 10 ms of audio**

> 🚀 **Coming Soon:** We are currently fine-tuning a new model optimized specifically for environments with even **louder background noise and louder background speech**! Stay tuned for the upcoming release.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/pulp-vision/Hush)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

---

## Model Overview

Hush is designed from the ground up for **Voice AI applications** — phone-based voice agents, call centre bots, voice assistants, real-time transcription pipelines, and conversational AI systems. It isolates exactly one speaker from a live audio stream, in real time, under production conditions.

The model is **language-agnostic** — it operates on the acoustic signal directly and works for any spoken language.

### At a Production Glance

| | |
|---|---|
| Model size | **8 MB** |
| Runs on | **CPU only — no GPU required** |
| Processing latency | **< 1 ms per 10 ms of audio** |
| Algorithmic latency | ~20 ms (fully causal, zero lookahead) |
| Training data | **10,000+ hours** of mixed speech, noise, and competing speakers |
| Sample rate | 16 kHz (telephony-native: G.711, WebRTC, SIP) |
| Language | **Any** (language-agnostic speech enhancement) |

---

## The Problem It Solves

Every major open-source speech enhancement model (DeepFilterNet3, RNNoise, SEGAN, MetricGAN+, DNS-Challenge entrants) is trained on **stationary noise** — fans, traffic, keyboard clicks. None treat a competing human voice as a first-class problem.

When the interference is another person speaking, these models either:
- **Leak the competing speaker** → gets transcribed as part of the conversation, breaking NLP/LLMs
- **Suppress both speakers** → degrades the primary speaker's intelligibility

**Hush is the first open-source model to explicitly train for background speaker suppression.**

---

## What Makes Hush Different

Built on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet), extended with one targeted innovation: **teaching the encoder to distinguish speakers, not just speech from noise.**

1. **Training data reflecting the real problem** — 60% of training samples include a competing human speaker at 12–24 dB SIR
2. **Auxiliary Separation Head** — lightweight `Linear(256→32) + Sigmoid` head trained with L1 loss to predict ERB-domain background speaker masks (training only — zero inference overhead)
3. **Joint optimization** — separation loss (weight 0.1) combined with multi-resolution spectral loss across 4 FFT scales

---

## Architecture

```
Input Waveform [B, 1, T]
        |
        v
  STFT (FFT=320, Hop=160)
        |
   _____|_______________
   |                   |
   v                   v
ERB features        DF features
[B, 1, T, 32]      [B, 2, T, 64]
   |                   |
   '-------+------------'
           |
           v
        ENCODER
   (SqueezedGRU, 256-dim)
           |
   ________|____________________________
   |               |                   |
   v               v                   v
ERB DECODER     DF DECODER     SEPARATION HEAD *
(ConvTranspose  (3-layer GRU   (Linear + Sigmoid
 + skip conns)   + DF filter)   ERB-domain mask)
   |               |
   v               v
ERB gain mask   Complex filter
   |               |
   '-------+--------'
           |
           v
    Enhanced Spectrum
           |
           v
         ISTFT
           |
           v
   Enhanced Waveform [B, 1, T]
```

`*` Separation Head is active during training only — discarded at inference.

### Model Specifications

| Parameter | Value |
|---|---|
| Model size | **8 MB** |
| Parameters | ~1.8M |
| Sample rate | 16,000 Hz |
| Frame size / hop | 320 / 160 samples (10 ms) |
| ERB bands | 32 |
| DF bins | 64 (order-5 filter) |
| Encoder dim | 256 |
| Lookahead | 0 (fully causal) |

---

## Quick Start: PyTorch Inference

```python
import torch
import soundfile as sf
from model.dfnet_se import DfNetSE, get_config

config = get_config()
model = DfNetSE(config)
checkpoint = torch.load("model_best.ckpt", map_location="cpu")
model.model.load_state_dict(checkpoint)
model.eval()

audio, sr = sf.read("noisy_speech.wav")
assert sr == 16000, "Input must be 16 kHz"

wav = torch.tensor(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
with torch.no_grad():
    enhanced = model(wav)  # [1, 1, T]

sf.write("enhanced.wav", enhanced.squeeze().numpy(), 16000)
```

## Quick Start: Production (ONNX, No PyTorch)

For production deployment without PyTorch, use the prebuilt **Weya NC Standalone** library:

```python
import ctypes, platform, numpy as np

lib_name = {"Darwin": "libweya_nc.dylib", "Windows": "weya_nc.dll"}.get(
    platform.system(), "libweya_nc.so"
)
lib = ctypes.CDLL(f"deployment/lib/{lib_name}")

model = lib.weya_nc_model_load_from_path(b"onnx/advanced_dfnet16k_model_best_onnx.tar.gz")
session = lib.weya_nc_session_create(model, 16000, ctypes.c_float(100.0))
frame_len = int(lib.weya_nc_get_frame_length(session))
lib.weya_nc_process_frame(session, input_ptr, output_ptr)
```

Prebuilt binaries are available for Linux, macOS (Apple Silicon), and Windows. See the [deployment guide](https://github.com/pulp-vision/Hush/tree/main/deployment) for full integration instructions.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 (cosine decay to 1e-6) |
| LR warmup | 3 epochs (1e-4 → 5e-4) |
| Weight decay | 0.05 |
| Batch size | 16 |
| Max sample length | 5 seconds |
| Epochs | 100 |
| Early stopping | patience=25 epochs |
| Gradient clip | 1.0 |
| Loss | MultiResSpecLoss (4 scales) + LocalSNRLoss + SeparationLoss (×0.1) |
| Background speaker prob. | 60% of samples |
| Background SIR range | 12–24 dB |

---

## Datasets Used

The model was trained on standard publicly available datasets totalling **over 10,000 hours of mixed audio**:

| Category | Datasets |
|---|---|
| **Primary speech** | LibriSpeech (train-clean-100/360), VCTK Corpus, Common Voice |
| **Background speech** | LibriSpeech / VCTK / LibriTTS (speaker-disjoint splits) |
| **Noise** | DNS Challenge, FreeSound, ESC-50, AudioSet |
| **Room impulse responses** | MIT IR Survey, OpenAIR, BUT ReverbDB |

> **Note:** Speech enhancement operates on acoustic features, not linguistic content — Hush works effectively across all languages.

See [DATASETS.md](https://github.com/pulp-vision/Hush/blob/main/DATASETS.md) for full details with URLs and licensing.

---

## Known Limitations

- **16 kHz only** — trained and evaluated at 16 kHz; other sample rates require resampling
- **Separation head is auxiliary** — the background speaker mask is an ERB-domain soft mask used for training regularization, not a standalone source separation output
- **Background speakers at moderate SIR** — trained with background speakers at 12–24 dB SIR; very loud competing speakers may not be fully suppressed

---

## Repository Structure

```
weya-ai/hush/  (this Hugging Face repo)
├── README.md                  ← This Model Card
├── config.json                ← Model configuration metadata
├── model_best.ckpt            ← PyTorch checkpoint
├── onnx/
│   └── advanced_dfnet16k_model_best_onnx.tar.gz  ← ONNX production bundle
└── LICENSE                    ← Apache 2.0
```

Full source code, training scripts, deployment examples, and documentation are available on [**GitHub**](https://github.com/pulp-vision/Hush).

---

## Acknowledgements

Built on [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B., and Andreas Maier. The core architecture, ERB filterbank, SqueezedGRU module, and loss functions closely follow the DF3 design.

---

## Citation

If you use this model or code, please cite the original DeepFilterNet paper:

```bibtex
@inproceedings{schroter2023deepfilternet3,
  title     = {DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement},
  author    = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N and Maier, Andreas},
  booktitle = {INTERSPEECH},
  year      = {2023}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
