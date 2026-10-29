"""Exercise the SLARDA DINOv3 adaptation model with synthetic EEG.

Run from the repository root. Random weights are used unless --pretrained
is passed with a locally available checkpoint.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("branch", choices=("adaptation",))
parser.add_argument("--pretrained", action="store_true", help="load the local Meta checkpoint")
options = parser.parse_args()
branch_dir = ROOT / "Autorgressive_Adaptation"
os.chdir(branch_dir)
sys.path.insert(0, str(branch_dir))
sys.argv = [sys.argv[0]]  # Upstream args.py parses at import time.

from config_files.EEG_Configs import Config  # noqa: E402
from models.models import Discriminator_AR, EEG_DINOv3_S  # noqa: E402

config = Config()
config.base_model.dinov3_pretrained = options.pretrained
model = EEG_DINOv3_S(config.base_model)

torch.manual_seed(7)
eeg = torch.randn(2, 1, 3000)
logits, (flat, temporal) = model(eeg)
assert logits.shape == (2, 5), logits.shape
assert flat.shape == (2, 128 * 22), flat.shape
assert temporal.shape == (2, 128, 22), temporal.shape
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
loss.backward()
assert model.feature_extractor.project.weight.grad is not None
assert any(p.grad is not None for p in model.feature_extractor.backbone.parameters())

discriminator = Discriminator_AR(config)
domain_logits = discriminator(temporal.detach())
assert domain_logits.shape == (2, 1), domain_logits.shape
domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    domain_logits, torch.tensor([[1.0], [0.0]])
)
domain_loss.backward()
assert discriminator.DC.weight.grad is not None
print("adaptation", "logits", tuple(logits.shape), "temporal", tuple(temporal.shape),
      "classification forward/backward OK")
