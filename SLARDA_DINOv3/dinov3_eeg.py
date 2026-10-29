"""Adapt 30-second, single-channel EEG epochs to DINOv3 ViT-S/16 tokens.

The adapter returns [batch, 128, 22], matching SLARDA's EEG CNN feature
contract. Only the encoder changes; the classifiers and training losses stay
in the original training branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
LOCAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "model.safetensors"
IMAGE_SIZE = 224
FEATURE_CHANNELS = 128
FEATURE_STEPS = 22


def _load_meta_safetensors(backbone, checkpoint_path):
    """Load Meta's raw DINOv3 ViT-S/16 keys into Transformers' module names."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Install safetensors to load a local DINOv3 checkpoint") from exc

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")
    raw = load_file(str(checkpoint_path), device="cpu")
    required = {"cls_token", "reg_token", "patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight", "norm.bias"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Not a raw Meta DINOv3 ViT-S/16 checkpoint; missing {sorted(missing)}")

    converted = {
        "embeddings.cls_token": raw["cls_token"],
        "embeddings.register_tokens": raw["reg_token"].unsqueeze(0) if raw["reg_token"].ndim == 2 else raw["reg_token"],
        "embeddings.patch_embeddings.weight": raw["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": raw["patch_embed.proj.bias"],
        "norm.weight": raw["norm.weight"],
        "norm.bias": raw["norm.bias"],
    }
    for index in range(12):
        prefix = f"blocks.{index}."
        target = f"model.layer.{index}."
        qkv = raw.get(prefix + "attn.qkv.weight")
        if qkv is None:
            raise ValueError(f"Checkpoint is missing {prefix}attn.qkv.weight")
        q_weight, k_weight, v_weight = qkv.chunk(3, dim=0)
        converted.update({
            target + "norm1.weight": raw[prefix + "norm1.weight"],
            target + "norm1.bias": raw[prefix + "norm1.bias"],
            target + "attention.q_proj.weight": q_weight,
            target + "attention.k_proj.weight": k_weight,
            target + "attention.v_proj.weight": v_weight,
            target + "attention.o_proj.weight": raw[prefix + "attn.proj.weight"],
            target + "attention.o_proj.bias": raw[prefix + "attn.proj.bias"],
            target + "layer_scale1.lambda1": raw[prefix + "gamma_1"],
            target + "norm2.weight": raw[prefix + "norm2.weight"],
            target + "norm2.bias": raw[prefix + "norm2.bias"],
            target + "mlp.up_proj.weight": raw[prefix + "mlp.fc1.weight"],
            target + "mlp.up_proj.bias": raw[prefix + "mlp.fc1.bias"],
            target + "mlp.down_proj.weight": raw[prefix + "mlp.fc2.weight"],
            target + "mlp.down_proj.bias": raw[prefix + "mlp.fc2.bias"],
            target + "layer_scale2.lambda1": raw[prefix + "gamma_2"],
        })

    incompatible = backbone.load_state_dict(converted, strict=False)
    allowed_missing = {"embeddings.mask_token"}
    unexpected = set(incompatible.unexpected_keys)
    actual_missing = set(incompatible.missing_keys)
    if unexpected or actual_missing.difference(allowed_missing):
        raise RuntimeError(
            "DINOv3 checkpoint conversion incomplete: "
            f"missing={sorted(actual_missing)}, unexpected={sorted(unexpected)}"
        )
    return checkpoint_path


class EEGToDINOv3(nn.Module):
    def __init__(self, *, model_id=MODEL_ID, pretrained=True):
        super().__init__()
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError as exc:
            raise ImportError("Install torch and a Transformers release with DINOv3 support") from exc

        if pretrained:
            checkpoint = Path(model_id).expanduser() if model_id else LOCAL_CHECKPOINT
            if checkpoint.is_file():
                self.backbone = DINOv3ViTModel(
                    DINOv3ViTConfig(
                        image_size=IMAGE_SIZE,
                        patch_size=16,
                        hidden_size=384,
                        intermediate_size=1536,
                        num_hidden_layers=12,
                        num_attention_heads=6,
                        num_register_tokens=4,
                        query_bias=False,
                        key_bias=False,
                        value_bias=False,
                        mlp_bias=True,
                        layerscale_value=1e-5,
                    )
                )
                _load_meta_safetensors(self.backbone, checkpoint)
            else:
                # A local directory from save_pretrained() also works. This
                # must fail loudly if access to gated weights is unavailable.
                self.backbone = DINOv3ViTModel.from_pretrained(model_id)
        else:
            # Explicitly for smoke tests. Never silently substitute random
            # weights for the pretrained DINOv3 checkpoint.
            self.backbone = DINOv3ViTModel(
                DINOv3ViTConfig(num_register_tokens=4)
            )

        cfg = self.backbone.config
        if cfg.hidden_size != 384 or cfg.patch_size != 16 or cfg.num_channels != 3:
            raise ValueError("Expected DINOv3 ViT-S/16 with 384-dim RGB patch tokens")
        self.project = nn.Linear(cfg.hidden_size, FEATURE_CHANNELS)
        self.register_buffer("window", torch.hann_window(256), persistent=False)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _spectrogram(self, eeg):
        if eeg.ndim != 3 or eeg.shape[1:] != (1, 3000):
            raise ValueError(f"Expected EEG [B, 1, 3000], got {tuple(eeg.shape)}")
        if not torch.is_floating_point(eeg):
            raise TypeError("EEG samples must be floating point")
        # STFT runs in float32 for reproducible behavior with mixed precision.
        wave = eeg[:, 0].float()
        spectrum = torch.stft(wave, n_fft=256, hop_length=64,
                              window=self.window.float(), return_complex=True)
        image = torch.log1p(spectrum.abs()).unsqueeze(1)
        image = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE),
                              mode="bilinear", align_corners=False)
        # Map each epoch's dynamic range to image-like values before using the
        # image backbone. Time remains the horizontal axis.
        low = image.amin(dim=(-2, -1), keepdim=True)
        high = image.amax(dim=(-2, -1), keepdim=True)
        image = (image - low) / (high - low).clamp_min(1e-6)
        image = image.repeat(1, 3, 1, 1)
        return (image - self.rgb_mean) / self.rgb_std

    def forward(self, eeg):
        image = self._spectrogram(eeg)
        tokens = self.backbone(pixel_values=image).last_hidden_state
        grid = IMAGE_SIZE // self.backbone.config.patch_size
        prefix = tokens.shape[1] - grid * grid
        if prefix != 1 + self.backbone.config.num_register_tokens:
            raise RuntimeError("Unexpected DINOv3 CLS/register/patch token layout")
        patches = tokens[:, prefix:].reshape(eeg.shape[0], grid, grid, -1)
        # Average frequency patches, retain left-to-right temporal ordering.
        timeline = patches.mean(dim=1)
        features = self.project(timeline).transpose(1, 2)
        return F.interpolate(features, size=FEATURE_STEPS, mode="linear", align_corners=False)
