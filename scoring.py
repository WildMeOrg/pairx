"""PairX re-ranking scores for MiewID.

Computes two discriminative metrics from PairX intermediate-layer matching:
  - Inverted Residual Mean (IRM): geometric consistency of feature correspondences
  - Match Coverage (MC): relevance-weighted quality of feature correspondences

These metrics provide spatial correspondence signals orthogonal to cosine
similarity and can separate correct from incorrect matches even when cosine
scores are tied.

Reference: Shrack et al., "Pairwise Matching of Intermediate Representations
for Fine-grained Explainability" (arXiv 2503.22881)
"""
import math
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import (
    choose_canonizer,
    get_feature_matches,
    get_intermediate_feature_maps_and_embedding,
    get_intermediate_relevances,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def pairx_rerank_score(device, img_0, img_1, model, layer_key,
                       w_irm=0.5, w_mc=0.5, mc_top_k=20):
    """Compute a PairX re-ranking score for an image pair.

    Runs PairX's intermediate-layer matching (forward + backward passes)
    and returns a composite score from IRM and MC metrics.  Skips the
    expensive pixel-level backpropagation used for visualization.

    Args:
        device: torch device
        img_0: query image tensor (1, C, H, W) with requires_grad=True
        img_1: candidate image tensor (1, C, H, W) with requires_grad=True
        model: MiewID model (must be in eval mode, .device must be set)
        layer_key: intermediate layer key (e.g. 'backbone.blocks.5')
        w_irm: weight for IRM in composite score (default 0.5)
        w_mc: weight for MC in composite score (default 0.5)
        mc_top_k: number of top matches for MC computation (default 20)

    Returns:
        float: composite PairX score, or None on failure.
    """
    try:
        layer_keys = [layer_key]

        # Forward pass: get intermediate feature maps and embeddings
        feature_maps_0, emb_0 = get_intermediate_feature_maps_and_embedding(
            img_0, model, layer_keys
        )
        feature_maps_1, emb_1 = get_intermediate_feature_maps_and_embedding(
            img_1, model, layer_keys
        )

        # Backprop cosine similarity to get intermediate relevances
        emb_0.retain_grad()
        emb_1.retain_grad()
        cosine_sim = F.cosine_similarity(emb_0, emb_1, dim=1)
        cosine_sim.backward()

        intermediate_relevances_0 = get_intermediate_relevances(
            img_0, emb_0.grad, model, layer_keys
        )
        intermediate_relevances_1 = get_intermediate_relevances(
            img_1, emb_1.grad, model, layer_keys
        )

        # Feature matching via BFMatcher cross-check
        # Remove batch dimension: [1, C, H, W] -> [C, H, W]
        # core.py's flatten_to_descriptors expects [C, H, W] not [1, C, H, W]
        # Use [0] indexing instead of squeeze() to fail loudly on unexpected shapes
        fm_0 = feature_maps_0[layer_key][0]
        fm_1 = feature_maps_1[layer_key][0]
        ir_0 = intermediate_relevances_0[layer_key][0]
        ir_1 = intermediate_relevances_1[layer_key][0]

        matches = get_feature_matches(fm_0, fm_1, img_0, img_1)

        if len(matches) == 0:
            return None

        # Compute relevance for each match
        for match in matches:
            i0, j0 = match['coord0']
            i1, j1 = match['coord1']
            match['relevance'] = float(ir_0[j0][i0] * ir_1[j1][i1])

        matches.sort(key=lambda x: -x['relevance'])

        # --- IRM: Inverted Residual Mean ---
        irm = _compute_irm(matches)

        # --- MC: Match Coverage ---
        mc = _compute_mc(matches, mc_top_k)

        # Composite score
        if irm is not None and mc is not None:
            score = w_irm * irm + w_mc * mc
        elif mc is not None:
            # Not enough matches for homography; MC-only fallback
            score = mc
        else:
            return None

        return float(score)

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            raise
        logger.warning('PairX scoring failed for pair', exc_info=True)
        return None
    except Exception:
        logger.warning('PairX scoring failed for pair', exc_info=True)
        return None


def _compute_irm(matches):
    """Inverted Residual Mean: geometric consistency via homography fit.

    Returns 1 / (1 + mean_residual), or None if < 4 matches.
    """
    if len(matches) < 4:
        return None

    src_pts = np.array(
        [m['keypoint0'].pt for m in matches], dtype=np.float32
    ).reshape(-1, 1, 2)
    dst_pts = np.array(
        [m['keypoint1'].pt for m in matches], dtype=np.float32
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    projected = cv2.perspectiveTransform(src_pts, H)
    residuals = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=2)).flatten()
    mean_residual = float(np.mean(residuals))

    return 1.0 / (1.0 + mean_residual)


def _compute_mc(matches, top_k=20):
    """Match Coverage: mean relevance of top-k matches."""
    if len(matches) == 0:
        return None

    k = min(top_k, len(matches))
    top_relevances = [m['relevance'] for m in matches[:k]]
    return float(np.mean(top_relevances))


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def normalize_pairx_score(raw_score, k=10.0, x0=0.5):
    """Sigmoid normalization of a raw PairX score to [0, 1].

    Same pattern as Hybrid plugin's LightGlue normalization.
    """
    exponent = -k * (raw_score - x0)
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))
