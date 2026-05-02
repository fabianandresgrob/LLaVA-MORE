"""
SAE Bottleneck module for LLaVA-MORE.

Wraps a pretrained BatchTopK Sparse Autoencoder (SAE) as a frozen bottleneck
layer inserted between the CLIP vision encoder and the MLP projection.
The SAE encode/decode is applied to each token independently.
"""

import os
import sys
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SAEBottleneck(nn.Module):
    """
    A frozen SAE bottleneck inserted between the CLIP vision encoder and the MLP projector.

    The SAE was trained on patch token activations from CLIP ViT-L/14-336 layer 22
    using the BatchTopK architecture from the dictionary_learning library.

    Two modes:
    - encode_only=False (default): encode then decode, output is reconstructed 1024d features
    - encode_only=True: encode only, output is sparse 8192d SAE activations
    """

    def __init__(self, sae_checkpoint_path: str, encode_only: bool = False,
                 log_stats: bool = True, log_interval: int = 100):
        super().__init__()
        self.encode_only = encode_only
        self.log_stats = log_stats
        self.log_interval = log_interval
        self._step_counter = 0

        self.sae = self._load_sae(sae_checkpoint_path)

        for param in self.sae.parameters():
            param.requires_grad = False
        self.sae.eval()

        try:
            k_str = str(self.sae.k.item())
            threshold_str = f"{self.sae.threshold.item():.6f}"
        except RuntimeError:
            k_str = threshold_str = "<meta>"
        logger.info(
            f"SAE Bottleneck initialized (encode_only={self.encode_only}): "
            f"activation_dim={self.sae.activation_dim}, "
            f"dict_size={self.sae.dict_size}, k={k_str}, "
            f"threshold={threshold_str}, "
            f"output_dim={self.output_dim}"
        )

    def _load_sae(self, checkpoint_path: str):
        # LLaVA-MORE layout: src/llava/model/sae_bottleneck.py
        # Go up 4 dirs to reach LLaVA-MORE root, then one more to GuidedResearch/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        llava_more_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
        sae_repo_path = os.path.join(os.path.dirname(llava_more_root), "sae-for-vlm")
        if os.path.isdir(sae_repo_path) and sae_repo_path not in sys.path:
            sys.path.insert(0, sae_repo_path)

        from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

        if os.path.isdir(checkpoint_path):
            ae_path = os.path.join(checkpoint_path, "ae.pt")
            if not os.path.exists(ae_path):
                raise FileNotFoundError(
                    f"Could not find ae.pt in {checkpoint_path}. "
                    f"Contents: {os.listdir(checkpoint_path)}"
                )
        else:
            ae_path = checkpoint_path

        logger.info(f"Loading SAE from {ae_path}")
        sae = BatchTopKSAE.from_pretrained(ae_path)
        for param in sae.parameters():
            param.data = param.data.contiguous()
        return sae

    @property
    def output_dim(self) -> int:
        if self.encode_only:
            return self.sae.dict_size
        return self.sae.activation_dim

    @torch.no_grad()
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, num_tokens, D] visual features from CLIP
        Returns:
            [B, num_tokens, dict_size] if encode_only else [B, num_tokens, D]
        """
        input_dtype = image_features.dtype
        B, N, D = image_features.shape

        sae_dtype = self.sae.encoder.weight.dtype
        x = image_features.to(sae_dtype)
        x_flat = x.reshape(B * N, D)

        x_hat_flat, encoded_acts = self.sae(x_flat, output_features=True)

        if self.log_stats and self.training:
            self._step_counter += 1
            if self._step_counter % self.log_interval == 0:
                self._log_sae_stats(x_flat, x_hat_flat, encoded_acts)

        if self.encode_only:
            out = encoded_acts.reshape(B, N, -1)
        else:
            out = x_hat_flat.reshape(B, N, D)

        return out.to(input_dtype)

    def _log_sae_stats(self, x: torch.Tensor, x_hat: torch.Tensor, encoded_acts: torch.Tensor):
        try:
            import wandb
            if not wandb.run:
                return

            recon_error = (x - x_hat).norm(dim=-1) / (x.norm(dim=-1) + 1e-8)
            active_per_token = (encoded_acts > 0).float().sum(dim=-1).mean().item()
            active_mask = encoded_acts > 0
            if active_mask.any():
                active_values = encoded_acts[active_mask]
                mean_activation = active_values.mean().item()
                max_activation = active_values.max().item()
            else:
                mean_activation = 0.0
                max_activation = 0.0

            x_centered = x - x.mean(dim=0, keepdim=True)
            total_var = x_centered.pow(2).sum()
            residual_var = (x - x_hat).pow(2).sum()
            fve = 1 - (residual_var / (total_var + 1e-8))

            wandb.log({
                "sae/reconstruction_error": recon_error.mean().item(),
                "sae/active_features_per_token": active_per_token,
                "sae/mean_activation": mean_activation,
                "sae/max_activation": max_activation,
                "sae/fve": fve.item(),
            }, commit=False)
        except Exception:
            pass

    def extra_repr(self) -> str:
        return (
            f"activation_dim={self.sae.activation_dim}, "
            f"dict_size={self.sae.dict_size}, "
            f"encode_only={self.encode_only}, "
            f"output_dim={self.output_dim}"
        )
