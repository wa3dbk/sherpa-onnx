"""Round-trip a wav through the exported Higgs-Audio-V2 ONNX codec.

    wav -> encoder.onnx -> codes -> decoder.onnx -> wav'

Reports token count, per-codebook value range, RMSE and SI-SDR between the
resampled input and the round-tripped output. This is the fastest way to
confirm the exported graphs actually work before wiring them into C++.

Usage:
    python -m omnivoice.scripts.roundtrip_onnx_smoketest \\
        --encoder hf/omnivoice-onnx/onnx2/higgs_codec_encoder.onnx \\
        --decoder hf/omnivoice-onnx/onnx2/higgs_codec_decoder.onnx \\
        --wav path/to/input.wav \\
        --out  round_trip.wav \\
        [--sample-rate 24000] [--providers cpu|cuda]
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


TARGET_SR = 24000


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        try:
            import librosa
        except ImportError as e:
            raise SystemExit(
                f"input is {sr} Hz, need {target_sr}; install librosa or "
                f"pre-resample the wav"
            ) from e
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return wav


def _si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    proj = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    noise = est - proj
    return 10.0 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))


def _providers(name: str) -> list:
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="round_trip.wav")
    ap.add_argument("--sample-rate", type=int, default=TARGET_SR)
    ap.add_argument("--providers", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    sr = args.sample_rate
    wav = _load_mono(args.wav, sr)
    print(f"input: {args.wav}  samples={len(wav)}  duration={len(wav)/sr:.2f}s  sr={sr}")

    x = wav[None, None, :].astype(np.float32)  # [1, 1, T]

    providers = _providers(args.providers)
    enc = ort.InferenceSession(args.encoder, providers=providers)
    dec = ort.InferenceSession(args.decoder, providers=providers)
    print(f"encoder providers: {enc.get_providers()}")

    codes = enc.run(["audio_codes"], {"input_values": x})[0]  # [1, 8, T_tok]
    print(f"codes: shape={codes.shape}  dtype={codes.dtype}  "
          f"min={int(codes.min())}  max={int(codes.max())}")
    for c in range(codes.shape[1]):
        cc = codes[0, c]
        print(f"  codebook {c}: min={int(cc.min())}  max={int(cc.max())}  "
              f"unique={len(np.unique(cc))}")

    frame_rate = codes.shape[2] / (len(wav) / sr)
    print(f"empirical frame rate: {frame_rate:.2f} tok/s  (expected 25)")

    y = dec.run(["audio_values"], {"audio_codes": codes.astype(np.int64)})[0]  # [1, 1, T']
    y = y.squeeze()
    print(f"reconstructed: samples={len(y)}  duration={len(y)/sr:.2f}s")

    n = min(len(wav), len(y))
    rmse = float(np.sqrt(np.mean((wav[:n] - y[:n]) ** 2)))
    sisdr = _si_sdr(wav[:n], y[:n])
    print(f"RMSE={rmse:.5f}  SI-SDR={sisdr:.2f} dB")

    out = Path(args.out)
    sf.write(str(out), y, sr, subtype="PCM_16")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
