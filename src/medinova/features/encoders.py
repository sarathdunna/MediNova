"""Frozen image and text encoders (proposal 6.1).

All heavy imports are lazy so the rest of the package works without them.
Install with `pip install -e .[encoders]`. Weights download on first use.

Week-5 checklist for each encoder: CPU seconds per item, output dim, and whether
negated findings ("no effusion" vs "small effusion") separate in embedding space
(see `negation_probe`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch
from PIL import Image


class ImageEncoder(Protocol):
    name: str
    dim: int

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


class TextEncoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


# --- image ---------------------------------------------------------------------


class TXVDenseNet:
    """torchxrayvision DenseNet-121 trained on MIMIC-CXR (CheXpert labels); 1024-d pooled features.

    The default excludes Open-I: the popular "densenet121-res224-all" weights were trained
    partly on Open-I, which is IU X-Ray itself, so they would leak test images into the encoder.
    Those weights stay available as `txv_densenet121_all`, for a disclosed comparison only.
    """

    name, dim = "txv_densenet121", 1024
    weights = "densenet121-res224-mimic_ch"

    def __init__(self):
        import sys

        import torchvision
        import torchxrayvision as xrv

        # torchxrayvision's download progress bar crashes on non-UTF-8 Windows consoles.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        self.xrv = xrv
        self.model = xrv.models.DenseNet(weights=self.weights).eval()
        self.transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        batch = []
        for img in images:
            x = np.asarray(img.convert("L"), dtype=np.float32)
            x = self.xrv.datasets.normalize(x, 255)[None, ...]  # (1, H, W) in [-1024, 1024]
            batch.append(torch.from_numpy(self.transform(x)))
        return self.model.features2(torch.stack(batch)).numpy()


class TXVDenseNetAll(TXVDenseNet):
    """Pooled-dataset weights INCLUDING Open-I (= IU X-Ray). Overlap ablation only; disclose it."""

    name, weights = "txv_densenet121_all", "densenet121-res224-all"


class BiomedCLIPImage:
    name, dim = "biomedclip_image", 512
    HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self):
        import open_clip

        self.model, self.preprocess = open_clip.create_model_from_pretrained(self.HUB)
        self.model.eval()

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(i.convert("RGB")) for i in images])
        return self.model.encode_image(x).numpy()


class MobileNetV3:
    """Generic ImageNet encoder; ablation only."""

    name, dim = "mobilenet_v3_small", 576

    def __init__(self):
        import torchvision

        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = torchvision.models.mobilenet_v3_small(weights=weights).eval()
        self.features = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten())
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(i.convert("RGB")) for i in images])
        return self.features(x).numpy()


# --- text ----------------------------------------------------------------------


class CXRBert:
    """Microsoft CXR-BERT-specialized; 128-d projected (contrastively trained) text embedding.

    The projection separates negated from affirmed findings far better than the raw [CLS]
    state (week-5 probe: "No pleural effusion" vs "Small left pleural effusion" cosine 0.29
    projected vs 0.77 [CLS]). The model runs remote code, so the revision is pinned.
    """

    name, dim = "cxr_bert", 128
    HUB = "microsoft/BiomedVLP-CXR-BERT-specialized"
    REVISION = "5157bdba1437a3aed316dacb1a5b68edf96b9902"

    def __init__(self, max_length: int = 256):
        from transformers import AutoModel, AutoTokenizer

        kw = {"trust_remote_code": True, "revision": self.REVISION}
        self.tok = AutoTokenizer.from_pretrained(self.HUB, **kw)
        self.model = AutoModel.from_pretrained(self.HUB, **kw).eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return self.model.get_projected_text_embeddings(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask
        ).numpy()


class BiomedCLIPText:
    name, dim = "biomedclip_text", 512

    def __init__(self):
        import open_clip

        self.model, _ = open_clip.create_model_from_pretrained(BiomedCLIPImage.HUB)
        self.tokenizer = open_clip.get_tokenizer(BiomedCLIPImage.HUB)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode_text(self.tokenizer(texts, context_length=256)).numpy()


class MiniLM:
    """Generic sentence transformer; ablation only."""

    name, dim = "minilm", 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, convert_to_numpy=True)


ENCODERS: dict[str, Callable[[], object]] = {
    "txv_densenet121": TXVDenseNet,
    "txv_densenet121_all": TXVDenseNetAll,
    "biomedclip_image": BiomedCLIPImage,
    "mobilenet_v3_small": MobileNetV3,
    "cxr_bert": CXRBert,
    "biomedclip_text": BiomedCLIPText,
    "minilm": MiniLM,
}


def get_encoder(name: str):
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    return ENCODERS[name]()


def negation_probe(encoder: TextEncoder) -> dict[str, float]:
    """Cosine similarity of negated vs affirmed findings; lower is better."""
    pairs = [
        ("No pleural effusion.", "Small left pleural effusion."),
        ("No pneumothorax.", "Large right pneumothorax."),
        ("Heart size is normal.", "The heart is enlarged."),
        ("No focal consolidation.", "Right lower lobe consolidation."),
    ]
    out = {}
    for a, b in pairs:
        ea, eb = encoder.encode([a, b])
        out[f"{a} | {b}"] = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)))
    return out
