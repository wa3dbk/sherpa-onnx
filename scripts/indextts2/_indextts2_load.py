# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Shared loaders for the IndexTTS-2 export scripts.

Each function returns a torch.nn.Module ready for `torch.onnx.export`.
Update these when the upstream repo changes its module layout. The
export scripts import from here so we have exactly one place to touch.
"""

from huggingface_hub import hf_hub_download


def _load_checkpoint(ckpt: str, variant: str):
    """Download the safetensors checkpoint file(s) for the variant."""
    return hf_hub_download(
        repo_id=ckpt, filename=f"{variant}/model.safetensors"
    )


def load_voice_encoder(ckpt: str, variant: str):
    from indextts2.models import VoiceEncoder  # upstream package
    ckpt_path = _load_checkpoint(ckpt, variant)
    model = VoiceEncoder.from_pretrained(ckpt_path)
    return model


def load_emotion_encoder(ckpt: str, variant: str):
    from indextts2.models import EmotionEncoder
    ckpt_path = _load_checkpoint(ckpt, variant)
    return EmotionEncoder.from_pretrained(ckpt_path)


def load_emotion_text_encoder(ckpt: str, variant: str):
    from indextts2.models import EmotionTextEncoder
    ckpt_path = _load_checkpoint(ckpt, variant)
    return EmotionTextEncoder.from_pretrained(ckpt_path)


def load_lm(ckpt: str, variant: str):
    from indextts2.models import IndexTTS2LM
    ckpt_path = _load_checkpoint(ckpt, variant)
    return IndexTTS2LM.from_pretrained(ckpt_path)


def load_vocoder(ckpt: str, variant: str):
    from indextts2.models import Vocoder
    ckpt_path = _load_checkpoint(ckpt, variant)
    return Vocoder.from_pretrained(ckpt_path)
