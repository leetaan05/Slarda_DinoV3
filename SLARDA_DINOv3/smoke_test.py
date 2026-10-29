"""Exercise real DINOv3 ViT-S/16 code with random weights and synthetic EEG.

Run each branch in its own process because the upstream branches both use
top-level modules named ``args`` and ``models``.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("branch", choices=("adaptation", "self_supervised"))
parser.add_argument("--pretrained", action="store_true", help="load the local Meta checkpoint")
options = parser.parse_args()
branch_dir = ROOT / ("Autorgressive_Adaptation" if options.branch == "adaptation" else "Self-supervised training")
os.chdir(branch_dir)
sys.path.insert(0, str(branch_dir))
sys.argv = [sys.argv[0]]  # Upstream args.py parses at import time.

from config_files.EEG_Configs import Config  # noqa: E402

config = Config()
if options.branch == "adaptation":
    config.base_model.dinov3_pretrained = options.pretrained
    from models.models import Discriminator_AR, EEG_DINOv3_S  # noqa: E402

    model = EEG_DINOv3_S(config.base_model)
else:
    from models.EEG_models import EEG_DINOv3_S  # noqa: E402

    config.dinov3_pretrained = options.pretrained
    config.training_mode = "self_supervised"
    model = EEG_DINOv3_S(config)

torch.manual_seed(7)
eeg = torch.randn(2, 1, 3000)
if options.branch == "self_supervised":
    cpc_loss = model(eeg)
    assert cpc_loss.ndim == 0 and torch.isfinite(cpc_loss)
    cpc_loss.backward()
    assert any(p.grad is not None for p in model.feature_extractor.backbone.parameters())
    print("CPC forward/backward:", float(cpc_loss.detach()))
    model.zero_grad(set_to_none=True)
    config.training_mode = "fine_tune"

logits, (flat, temporal) = model(eeg)
assert logits.shape == (2, 5), logits.shape
assert flat.shape == (2, 128 * 22), flat.shape
assert temporal.shape == (2, 128, 22), temporal.shape
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
loss.backward()
assert model.feature_extractor.project.weight.grad is not None
assert any(p.grad is not None for p in model.feature_extractor.backbone.parameters())
if options.branch == "adaptation":
    discriminator = Discriminator_AR(config)
    domain_logits = discriminator(temporal.detach())
    assert domain_logits.shape == (2, 1), domain_logits.shape
    domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        domain_logits, torch.tensor([[1.0], [0.0]])
    )
    domain_loss.backward()
    assert discriminator.DC.weight.grad is not None
print(options.branch, "logits", tuple(logits.shape), "temporal", tuple(temporal.shape),
      "classification forward/backward OK")
