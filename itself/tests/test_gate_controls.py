"""Focused CPU tests for reviewer-requested GATE attribution controls."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.enrichment.enricher import TargetPrototypeEnricher
from utils.gallery_controls import shuffle_gallery_evidence, similarity_weighted_qe_scores


def _args(variant):
    return SimpleNamespace(
        top_m=3,
        gate_variant=variant,
        enrich_gamma=0.1,
        residual_gate="static",
        tau=0.015,
        lambda_ret=0.5,
        context_module="mixer",
        enrichment_space="global",
        topm_rank_space="host_global",
        topm_rank_lambda=0.5,
        only_global=True,
        extractor_mode="global",
        num_parts=1,
        mixer_dim=8,
        mixer_depth=1,
        mixer_hidden_part=4,
        mixer_hidden_rank=5,
        mixer_hidden_channel=16,
        mixer_hidden_readout=8,
        context_pooling="mlp",
    )


class QueryOnlyTest(unittest.TestCase):
    def test_exact_same_parameters_and_gallery_independence(self):
        torch.manual_seed(17)
        full = TargetPrototypeEnricher(16, 32, _args("full")).eval()
        blind = TargetPrototypeEnricher(16, 32, _args("query_only")).eval()
        self.assertEqual(
            {k: tuple(p.shape) for k, p in full.named_parameters()},
            {k: tuple(p.shape) for k, p in blind.named_parameters()},
        )
        # Can load SAME state dict shape: the only change is information access.
        blind.load_state_dict(full.state_dict(), strict=True)
        q = torch.randn(2, 16)
        result_a = blind.enrich_only(q, q, pool_cache=None, space="global")
        result_b = blind.enrich_only(q, q, pool_cache={"arbitrary": torch.randn(9, 13)},
                                     space="global")
        self.assertTrue(torch.allclose(result_a, result_b, atol=0, rtol=0))

    def test_grab_uses_query_derived_projection(self):
        config = _args("query_only")
        config.enrichment_space = "grab"
        config.only_global = False
        blind = TargetPrototypeEnricher(16, 32, config).train()
        q_grab = torch.randn(2, 32)
        q_global = torch.randn(2, 16)
        a = blind.enrich_only(q_grab, q_global, None, "grab")
        b = blind.enrich_only(q_grab, q_global, {"any_gallery": True}, "grab")
        self.assertTrue(torch.equal(a, b))
        a.sum().backward()
        self.assertIsNotNone(blind.proto_to_grab.weight.grad)

    def test_training_uses_only_source_labels_for_loss(self):
        blind = TargetPrototypeEnricher(16, 32, _args("query_only")).train()
        q = torch.randn(2, 16, requires_grad=True)
        pool = {
            "retrieval_features": torch.randn(4, 16),
            "pids": torch.tensor([1, 2, 3, 1]),
        }
        result = blind(q, q, torch.tensor([1, 2]), pool, "global")
        self.assertTrue(torch.isfinite(result["total_loss"]).item())
        result["total_loss"].backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNone(result["top_indices"])


class EvidenceShuffleTest(unittest.TestCase):
    def test_shuffles_only_evidence_and_is_deterministic(self):
        n = 12
        cache = {
            "host_image_features": torch.randn(n, 4),
            "retrieval_features": torch.randn(n, 4),
            "pids": torch.arange(n),
            "top_indices": torch.arange(6).view(2, 3),
            "evidence_bank": torch.arange(n * 8).reshape(n, 2, 4).float(),
            "cluster_density_scalar": torch.arange(n).float().unsqueeze(1),
        }
        cache["prototypes"] = cache["evidence_bank"]
        a = shuffle_gallery_evidence(cache, 19)
        b = shuffle_gallery_evidence(cache, 19)
        self.assertTrue(torch.equal(a["evidence_bank"], b["evidence_bank"]))
        self.assertFalse(torch.equal(a["evidence_bank"], cache["evidence_bank"]))
        self.assertTrue(torch.equal(a["retrieval_features"], cache["retrieval_features"]))
        self.assertTrue(torch.equal(a["host_image_features"], cache["host_image_features"]))
        self.assertTrue(torch.equal(a["pids"], cache["pids"]))
        self.assertTrue(torch.equal(a["top_indices"], cache["top_indices"]))
        self.assertTrue(torch.equal(a["prototypes"], a["evidence_bank"]))
        self.assertTrue(torch.equal(a["evidence_bank"].flatten().sort().values,
                                    cache["evidence_bank"].flatten().sort().values))
        # No mutation of source cache.
        self.assertTrue(torch.equal(cache["evidence_bank"],
                                    torch.arange(n * 8).reshape(n, 2, 4).float()))


class QEControlTest(unittest.TestCase):
    def test_alpha_zero_reproduces_base_and_scores_full_gallery(self):
        q = torch.tensor([[1., 0.], [0., 1.]])
        g = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        rank_scores = F.normalize(q, dim=-1) @ F.normalize(g, dim=-1).t()
        scores = similarity_weighted_qe_scores(q, g, rank_scores, top_m=2, alpha=0)
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertTrue(torch.allclose(scores, rank_scores, atol=1e-6))

    def test_qe_uses_rank_space_but_context_vectors_in_query_space(self):
        q = torch.tensor([[1., 0.]])
        g = torch.tensor([[1., 0.], [0., 1.]])
        alternative_ranks = torch.tensor([[0., 10.]])
        scores = similarity_weighted_qe_scores(q, g, alternative_ranks,
                                               top_m=1, alpha=1., temperature=0.05)
        self.assertGreater(scores[0, 1], 0.)
        self.assertAlmostEqual(scores[0, 0].item(), scores[0, 1].item(), places=5)
        with self.assertRaises(ValueError):
            similarity_weighted_qe_scores(q, g, alternative_ranks,
                                          top_m=1, temperature=0.)


if __name__ == "__main__":
    unittest.main()
