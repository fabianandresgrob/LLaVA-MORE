#!/usr/bin/env python3
"""
Smoke test for the CLIP + SAE + Qwen3 pipeline.

Tiered tests — each tier requires more dependencies:
  Tier 0 (always): file structure and syntax checks — works on Mac with no env
  Tier 1 (torch+transformers): imports, template, SAE shapes — works on login node
  Tier 2 (real SAE checkpoint): live SAE load + forward — needs ae.pt on disk

Usage:
    # Mac, no env needed:
    python3 scripts/test_pipeline.py

    # Login node (torch available):
    python scripts/test_pipeline.py

    # With real SAE checkpoint:
    python scripts/test_pipeline.py --sae-checkpoint /path/to/ae.pt
"""

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results = []


def run_test(name, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        print(f"  [{PASS}] {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        print(f"          {type(e).__name__}: {e}")
        results.append((name, False))


def skip_test(name, reason):
    print(f"  [{SKIP}] {name}")
    print(f"          {reason}")
    results.append((name, None))


# ---------------------------------------------------------------------------
# Tier 0: pure Python — no torch/transformers needed
# ---------------------------------------------------------------------------

EXPECTED_SCRIPTS = [
    "scripts/llava-more/qwen3/pretrain.sh",
    "scripts/llava-more/qwen3/finetune.sh",
    "scripts/llava-more/qwen3/test.sh",
    "scripts/llava-more/qwen3/launch_sweep.sh",
    "src/llava/model/sae_bottleneck.py",
    "src/llava/model/language_model/llava_qwen3.py",
]

EXPECTED_PYTHON = [
    "src/llava/model/sae_bottleneck.py",
    "src/llava/model/language_model/llava_qwen3.py",
    "src/llava/model/llava_arch.py",
    "src/llava/conversation.py",
    "src/llava/load_utils.py",
]


def test_file_structure():
    missing = [p for p in EXPECTED_SCRIPTS if not (REPO_ROOT / p).exists()]
    assert not missing, f"Missing files: {missing}"


def test_python_syntax():
    errors = []
    for rel in EXPECTED_PYTHON:
        path = REPO_ROOT / rel
        try:
            ast.parse(path.read_text())
        except SyntaxError as e:
            errors.append(f"{rel}: {e}")
    assert not errors, "\n".join(errors)


def test_qwen3_template_raw():
    conv_path = REPO_ROOT / "src/llava/conversation.py"
    src = conv_path.read_text()
    assert '"qwen3"' in src or "'qwen3'" in src, "qwen3 key not found in conversation.py"
    # Ensure no think tags leaked into the training template
    assert "<think>" not in src, "<think> tags found in conversation.py — thinking mode must not appear in training template"


def test_sae_args_in_pretrain():
    pretrain = (REPO_ROOT / "scripts/llava-more/qwen3/pretrain.sh").read_text()
    assert "use_sae_bottleneck" in pretrain
    assert "sae_encode_only" in pretrain
    assert "--sae-enconly" in pretrain
    assert "--sae-encdec" in pretrain
    # Each SAE gets its own projector — run name must include dataset
    assert "${SAE_DATASET}" in pretrain, "pretrain run name must be SAE-specific"


def test_sae_args_in_finetune():
    finetune = (REPO_ROOT / "scripts/llava-more/qwen3/finetune.sh").read_text()
    assert "--sae-enconly" in finetune
    assert "--sae-encdec" in finetune
    assert "sae_encode_only False" in finetune, "enc+dec mode (encode_only=False) not found"
    assert "sae_encode_only True" in finetune, "enc-only mode (encode_only=True) not found"
    # Projector path must be SAE-specific — not the shared baseline path
    assert "pretrain-${SAE_DATASET}" in finetune, \
        "finetune must load SAE-specific pretrain projector, not shared baseline"


def test_launch_sweep_structure():
    launcher = (REPO_ROOT / "scripts/llava-more/qwen3/launch_sweep.sh").read_text()
    assert "afterok" in launcher, "dependency chain missing from sweep launcher"
    assert "enconly" in launcher
    assert "encdec" in launcher
    # Both modes should appear for each dataset
    assert launcher.count("pair enconly") >= 2, "expected pair enconly for each SAE"
    assert launcher.count("pair encdec") >= 2, "expected pair encdec for each SAE"


# ---------------------------------------------------------------------------
# Tier 1: requires torch + transformers
# ---------------------------------------------------------------------------

def _make_mock_sae(activation_dim=1024, dict_size=8192, k=20):
    import torch
    import torch.nn as nn

    class MockSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation_dim = activation_dim
            self.dict_size = dict_size
            self.k = torch.tensor(k)
            self.threshold = torch.tensor(0.0)
            self.encoder = nn.Linear(activation_dim, dict_size, bias=False)

        def forward(self, x, output_features=False):
            logits = self.encoder(x)
            topk_vals, topk_idx = torch.topk(logits, self.k.item(), dim=-1)
            acts = torch.zeros_like(logits)
            acts.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
            x_hat = acts @ self.encoder.weight
            if output_features:
                return x_hat, acts
            return x_hat

    return MockSAE()


def test_imports():
    from src.llava.model.sae_bottleneck import SAEBottleneck  # noqa: F401
    from src.llava.model.language_model.llava_qwen3 import LlavaQwen3Config  # noqa: F401
    from src.llava.conversation import conv_templates  # noqa: F401


def test_qwen3_conv_template():
    from src.llava.conversation import conv_templates
    assert "qwen3" in conv_templates
    conv = conv_templates["qwen3"].copy()
    conv.append_message(conv.roles[0], "Describe this image.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    assert isinstance(prompt, str) and len(prompt) > 0


def test_sae_bottleneck_encdec():
    import torch
    from unittest.mock import patch
    from src.llava.model.sae_bottleneck import SAEBottleneck

    mock = _make_mock_sae()
    with patch.object(SAEBottleneck, "_load_sae", return_value=mock):
        bn = SAEBottleneck.__new__(SAEBottleneck)
        bn.encode_only = False
        bn.log_stats = False
        bn._step_counter = 0
        bn.sae = mock
        for p in mock.parameters():
            p.requires_grad = False

        x = torch.randn(2, 576, 1024)
        out = bn(x)
        assert out.shape == (2, 576, 1024), f"wrong shape: {out.shape}"
        assert bn.output_dim == 1024


def test_sae_bottleneck_enconly():
    import torch
    from unittest.mock import patch
    from src.llava.model.sae_bottleneck import SAEBottleneck

    mock = _make_mock_sae()
    with patch.object(SAEBottleneck, "_load_sae", return_value=mock):
        bn = SAEBottleneck.__new__(SAEBottleneck)
        bn.encode_only = True
        bn.log_stats = False
        bn._step_counter = 0
        bn.sae = mock
        for p in mock.parameters():
            p.requires_grad = False

        x = torch.randn(2, 576, 1024)
        out = bn(x)
        assert out.shape == (2, 576, 8192), f"wrong shape: {out.shape}"
        assert bn.output_dim == 8192


# ---------------------------------------------------------------------------
# Tier 2: real SAE checkpoint
# ---------------------------------------------------------------------------

def test_real_sae_checkpoint(sae_path: str):
    import torch
    from src.llava.model.sae_bottleneck import SAEBottleneck

    for mode_name, encode_only in [("enc+dec", False), ("enc-only", True)]:
        bn = SAEBottleneck(sae_path, encode_only=encode_only, log_stats=False)
        x = torch.randn(2, 576, bn.sae.activation_dim)
        out = bn(x)
        expected = bn.sae.dict_size if encode_only else bn.sae.activation_dim
        assert out.shape == (2, 576, expected), f"{mode_name}: {out.shape}"
        assert not any(p.requires_grad for p in bn.sae.parameters()), "SAE not frozen"

    print(f"          dim={bn.sae.activation_dim}  dict={bn.sae.dict_size}  k={bn.sae.k.item()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-checkpoint", metavar="PATH")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    print("\nLLaVA-MORE pipeline smoke test")
    print("=" * 52)

    print("\n[Tier 0] File structure & syntax (no deps required)")
    run_test("expected files exist", test_file_structure)
    run_test("Python syntax valid", test_python_syntax)
    run_test("qwen3 template in conversation.py", test_qwen3_template_raw)
    run_test("SAE args wired in pretrain.sh", test_sae_args_in_pretrain)
    run_test("SAE modes wired in finetune.sh", test_sae_args_in_finetune)
    run_test("sweep launcher has dependency chain", test_launch_sweep_structure)

    print("\n[Tier 1] Imports & model logic (requires torch + transformers)")
    if has_torch:
        run_test("module imports", test_imports)
        run_test("qwen3 conversation template", test_qwen3_conv_template)
        run_test("SAEBottleneck enc+dec shapes", test_sae_bottleneck_encdec)
        run_test("SAEBottleneck enc-only shapes", test_sae_bottleneck_enconly)
    else:
        msg = "torch/transformers not installed — run on login node"
        for name in ["module imports", "qwen3 conversation template",
                     "SAEBottleneck enc+dec shapes", "SAEBottleneck enc-only shapes"]:
            skip_test(name, msg)

    print("\n[Tier 2] Real SAE checkpoint (requires ae.pt)")
    if args.sae_checkpoint:
        run_test("real SAE load + forward (both modes)", test_real_sae_checkpoint,
                 args.sae_checkpoint)
    else:
        skip_test("real SAE load + forward", "pass --sae-checkpoint PATH to enable")

    print("\n" + "=" * 52)
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    print(f"Results: {passed} passed  {failed} failed  {skipped} skipped\n")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
