"""Label-free inference controls for GATE (no test-time optimization)."""

import torch
import torch.nn.functional as F


_EVIDENCE_KEYS = (
    "evidence_bank",
    "prototypes",
    "retrieval_backbone_features",
    "cluster_features",
    "cluster_residual_features",
    "cluster_density_scalar",
    "cluster_rarity_scalar",
)


def shuffle_gallery_evidence(cache, seed):
    """Permute gallery-associated evidence only, preserving retrieval/index/IDs.

    One seeded permutation is shared across all queries. Images, retrieval features,
    initial Top-M indices and PID annotations are deliberately NOT permuted.
    This is an inference perturbation, not a parameter-matched capacity control.
    """
    if "evidence_bank" not in cache and "prototypes" not in cache:
        raise ValueError("Gallery evidence cache is missing evidence_bank/prototypes")
    n = int(cache["host_image_features"].shape[0])
    if n < 2:
        raise ValueError("Evidence shuffle requires at least two gallery images")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(n, generator=generator)
    if torch.equal(permutation, torch.arange(n)):
        permutation = permutation.roll(1)

    out = dict(cache)
    for name in _EVIDENCE_KEYS:
        value = cache.get(name)
        if value is None:
            continue
        if value.shape[0] != n:
            raise ValueError("Evidence field {} does not align with gallery".format(name))
        out[name] = value.index_select(0, permutation.to(value.device))
    # Keep the alias consistent when the provider uses both names.
    if "evidence_bank" in out and "prototypes" in out:
        out["prototypes"] = out["evidence_bank"]
    return out


def similarity_weighted_qe_scores(query_features, gallery_features, rank_scores,
                                  top_m, alpha=0.5, temperature=0.05,
                                  query_batch_size=64):
    """Return full-gallery scores after zero-training similarity-weighted QE.

    The ranking matrix can come from host_global, GRAB or the host's hybrid
    ranking policy. Expansion vectors are always in the SAME space as the
    chosen query representation. Scores retain the complete target gallery.
    """
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("QE features must have shape [N, D]")
    nq, dim = query_features.shape
    ng, gallery_dim = gallery_features.shape
    if dim != gallery_dim or ng == 0 or nq == 0:
        raise ValueError("QE requires nonempty compatible text/image features")
    if tuple(rank_scores.shape) != (nq, ng):
        raise ValueError("rank_scores must have shape [num_queries, num_gallery]")
    if int(top_m) <= 0 or int(query_batch_size) <= 0:
        raise ValueError("top_m and query_batch_size must be positive")
    if float(temperature) <= 0 or float(alpha) < 0:
        raise ValueError("QE temperature must be positive and alpha nonnegative")

    q = F.normalize(query_features.float(), p=2, dim=-1)
    g = F.normalize(gallery_features.float(), p=2, dim=-1)
    k = min(int(top_m), ng)
    outputs = []
    for start in range(0, nq, int(query_batch_size)):
        stop = min(start + int(query_batch_size), nq)
        values, indices = rank_scores[start:stop].float().topk(k, dim=1, sorted=True)
        weights = torch.softmax(values / float(temperature), dim=1)
        evidence = g[indices.to(g.device)]
        context = (evidence * weights.to(g.device).unsqueeze(-1)).sum(dim=1)
        enriched = F.normalize(q[start:stop].to(g.device) + float(alpha) * context,
                               p=2, dim=-1)
        outputs.append((enriched @ g.t()).cpu())
    return torch.cat(outputs, dim=0)
