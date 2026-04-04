#!/usr/bin/env python3
"""ONNX-only inference for Hush speech enhancement. No PyTorch required.

Uses onnxruntime for neural network inference and DeepFilterLib (libdf) for DSP.
DeepFilterLib is a lightweight Rust library with Python bindings — no PyTorch dependency.

Requirements:
    pip install onnxruntime DeepFilterLib numpy soundfile

Usage:
    python scripts/infer_onnx.py \
        --input noisy.wav \
        --output enhanced.wav

    # With explicit model directory:
    python scripts/infer_onnx.py \
        --model-dir deployment/models/onnx \
        --input noisy.wav \
        --output enhanced.wav
"""

from __future__ import annotations

import argparse
import math
import os
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from libdf import DF, erb, erb_norm, unit_norm
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants (from configs/run_config.json)
# ---------------------------------------------------------------------------
SR = 16000
FFT_SIZE = 320
HOP_SIZE = 160
NB_ERB = 32
NB_DF = 64
DF_ORDER = 5
DF_LOOKAHEAD = 0
NORM_TAU = 1.0
N_FREQS = FFT_SIZE // 2 + 1  # 161


def _norm_alpha() -> float:
    return math.exp(-HOP_SIZE / SR / NORM_TAU)


# ---------------------------------------------------------------------------
# Post-processing (applied after ONNX model outputs)
# ---------------------------------------------------------------------------

def apply_erb_mask(spec: NDArray, mask: NDArray, erb_inv_fb: NDArray) -> NDArray:
    """Apply ERB-domain mask to spectrum.

    Parameters
    ----------
    spec : complex64 [T, N_FREQS]
    mask : float32 [1, 1, T, NB_ERB] — sigmoid output from erb_dec
    erb_inv_fb : float32 [NB_ERB, N_FREQS]

    Returns
    -------
    complex64 [T, N_FREQS]
    """
    m = mask[0, 0]  # [T, 32]
    m_full = m @ erb_inv_fb  # [T, 161]
    return spec * m_full


def apply_df_filter(spec: NDArray, coefs: NDArray) -> NDArray:
    """Apply deep filter (5-tap complex FIR) to first NB_DF bins.

    Matches model/dfnet_se.py DfNet.apply_df exactly:
    - DfOutputReshapeMF: [B,T,F,O*2] → view [B,T,F,O,2] → permute(0,3,1,2,4) → [B,O,T,F,2]
    - apply_df: coefs.permute(0,2,1,3,4) → [B,T,O,F,2], complex multiply-accumulate

    Parameters
    ----------
    spec : complex64 [T, N_FREQS]
    coefs : float32 [1, T, NB_DF, DF_ORDER*2] from df_dec

    Returns
    -------
    complex64 [T, NB_DF]
    """
    t = spec.shape[0]

    # AIDEV-NOTE: Reshape chain matching DfOutputReshapeMF + apply_df permutations.
    # Net effect on unbatched: [T,64,10] → [T,64,5,2] → transpose to [T,5,64,2]
    c = coefs[0].reshape(t, NB_DF, DF_ORDER, 2)  # [T, F, O, 2]
    c = c.transpose(0, 2, 1, 3)  # [T, O, F, 2]
    c_re = c[..., 0]
    c_im = c[..., 1]

    # Pad spectrum: df_order-1 frames on left (lookahead=0)
    spec_df = spec[:, :NB_DF]
    spec_df_re = spec_df.real.astype(np.float32)
    spec_df_im = spec_df.imag.astype(np.float32)
    pad_left = DF_ORDER - 1 - DF_LOOKAHEAD  # 4
    spec_re_pad = np.pad(spec_df_re, ((pad_left, DF_LOOKAHEAD), (0, 0)))
    spec_im_pad = np.pad(spec_df_im, ((pad_left, DF_LOOKAHEAD), (0, 0)))

    # Sliding window of size DF_ORDER along time → [T, O, F]
    padded_re = np.stack([spec_re_pad[i:i + DF_ORDER] for i in range(t)], axis=0)
    padded_im = np.stack([spec_im_pad[i:i + DF_ORDER] for i in range(t)], axis=0)

    # Complex multiply + sum over order
    out_re = (padded_re * c_re - padded_im * c_im).sum(axis=1)
    out_im = (padded_im * c_re + padded_re * c_im).sum(axis=1)

    return (out_re + 1j * out_im).astype(np.complex64)


def build_erb_inv_fb(widths: NDArray) -> NDArray:
    """Build inverse ERB filterbank [NB_ERB, N_FREQS] matching model/dfnet_se.py erb_fb(inverse=True)."""
    n_freqs = int(widths.sum())
    fb = np.zeros((len(widths), n_freqs), dtype=np.float32)
    offset = 0
    for i, w in enumerate(widths):
        fb[i, offset:offset + w] = 1.0
        offset += w
    return fb


# ---------------------------------------------------------------------------
# ONNX model wrapper
# ---------------------------------------------------------------------------

class HushONNX:
    """Hush inference using ONNX Runtime + libdf. No PyTorch required."""

    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = ["CPUExecutionProvider"]

        self.enc = ort.InferenceSession(
            str(model_dir / "enc.onnx"), opts, providers=providers
        )
        self.erb_dec = ort.InferenceSession(
            str(model_dir / "erb_dec.onnx"), opts, providers=providers
        )
        self.df_dec = ort.InferenceSession(
            str(model_dir / "df_dec.onnx"), opts, providers=providers
        )

        # libdf handles STFT/ISTFT and feature extraction
        self.df_state = DF(
            sr=SR, fft_size=FFT_SIZE, hop_size=HOP_SIZE,
            nb_bands=NB_ERB, min_nb_erb_freqs=2,
        )
        self.alpha = _norm_alpha()
        self.erb_widths = self.df_state.erb_widths()
        self.erb_inv_fb = build_erb_inv_fb(self.erb_widths)

    def enhance(
        self,
        audio: NDArray,
        pad_delay: bool = True,
        atten_lim_db: float | None = None,
    ) -> NDArray:
        """Enhance a mono 16kHz float32 audio signal.

        Parameters
        ----------
        audio : float32 [T_samples]
        pad_delay : compensate for algorithmic delay
        atten_lim_db : max attenuation in dB (None = unlimited)

        Returns
        -------
        float32 [T_samples]
        """
        orig_len = len(audio)

        if pad_delay:
            audio = np.pad(audio, (0, FFT_SIZE), mode="constant")

        # --- STFT via libdf ---
        audio_2d = audio[np.newaxis, :]  # [1, T] for libdf
        spec_np = self.df_state.analysis(audio_2d, reset=True)  # [1, T_frames, N_FREQS] complex
        spec = spec_np[0]  # [T, N_FREQS]

        # --- Feature extraction via libdf ---
        erb_feat = erb_norm(erb(spec_np, self.erb_widths), self.alpha)  # [1, T, 32]
        erb_feat = erb_feat[:, np.newaxis, :, :].astype(np.float32)  # [1, 1, T, 32]

        spec_feat = unit_norm(spec_np[..., :NB_DF], self.alpha)  # [1, T, 64] complex
        # Convert to [1, 2, T, 64] (real, imag channels)
        spec_feat_ri = np.stack([spec_feat.real, spec_feat.imag], axis=1).astype(np.float32)

        # --- Encoder ---
        enc_out = self.enc.run(
            None,
            {"feat_erb": erb_feat, "feat_spec": spec_feat_ri},
        )
        enc_names = [o.name for o in self.enc.get_outputs()]
        enc_dict = dict(zip(enc_names, enc_out))

        emb = enc_dict["emb"]
        e0, e1, e2, e3 = enc_dict["e0"], enc_dict["e1"], enc_dict["e2"], enc_dict["e3"]
        c0 = enc_dict["c0"]

        # --- ERB Decoder ---
        mask = self.erb_dec.run(
            None,
            {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0},
        )[0]  # [1, 1, T, 32]

        # --- DF Decoder ---
        df_coefs = self.df_dec.run(
            None,
            {"emb": emb, "c0": c0},
        )[0]  # [1, T, 64, 10]

        # --- Apply ERB mask (all bins) ---
        spec_masked = apply_erb_mask(spec, mask, self.erb_inv_fb)

        # --- Apply DF filter (bins 0-63, on original spectrum) ---
        spec_df_enhanced = apply_df_filter(spec, df_coefs)

        # --- Merge: DF for bins 0-63, ERB masked for bins 64-160 ---
        spec_enhanced = spec_masked.copy()
        spec_enhanced[:, :NB_DF] = spec_df_enhanced

        # --- Attenuation limiting ---
        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / 20.0)
            spec_enhanced = spec * lim + spec_enhanced * (1.0 - lim)

        # --- ISTFT via libdf ---
        spec_enhanced_3d = spec_enhanced[np.newaxis, :, :]  # [1, T, N_FREQS]
        enhanced = self.df_state.synthesis(spec_enhanced_3d, reset=True)
        enhanced = np.asarray(enhanced[0], dtype=np.float32)

        # --- Delay compensation ---
        if pad_delay:
            delay = FFT_SIZE - HOP_SIZE  # 160 samples
            enhanced = enhanced[delay:orig_len + delay]

        return enhanced


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def extract_onnx_bundle(tar_path: Path, dest_dir: Path) -> Path:
    """Extract ONNX tar.gz bundle if not already extracted."""
    if (dest_dir / "enc.onnx").exists():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Denoise audio using Hush ONNX models (no PyTorch required)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input noisy WAV file")
    parser.add_argument("--output", type=Path, required=True, help="Output enhanced WAV file")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing enc.onnx, erb_dec.onnx, df_dec.onnx "
        "(default: auto-extract from deployment/models/)",
    )
    parser.add_argument(
        "--atten-lim-db",
        type=float,
        default=None,
        help="Max attenuation in dB (default: None = unlimited)",
    )
    parser.add_argument(
        "--no-delay-compensation",
        action="store_true",
        help="Skip delay compensation",
    )
    args = parser.parse_args()

    # Resolve model directory
    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        project_root = Path(__file__).resolve().parents[1]
        tar_path = project_root / "deployment" / "models" / "advanced_dfnet16k_model_best_onnx.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(
                f"ONNX bundle not found at {tar_path}. "
                "Use --model-dir to specify the directory with .onnx files."
            )
        model_dir = project_root / "deployment" / "models" / "onnx"
        extract_onnx_bundle(tar_path, model_dir)

    print(f"Loading ONNX models from {model_dir}")
    model = HushONNX(model_dir)

    # Load audio
    audio, sr = sf.read(str(args.input), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SR:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SR, sr).astype(np.float32)
        print(f"Resampled {sr} Hz -> {SR} Hz")

    duration = len(audio) / SR
    print(f"Input: {args.input} ({duration:.2f}s, {SR} Hz)")

    enhanced = model.enhance(
        audio,
        pad_delay=not args.no_delay_compensation,
        atten_lim_db=args.atten_lim_db,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), enhanced, SR)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
