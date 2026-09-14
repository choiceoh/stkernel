"""Proposal cost accounting and exact greedy tree semantics, independent of GPU timing."""
import unittest

import torch

from engine.modules.speculative_tree import Candidate, RouteTable, Tree, dflash_candidates, select
from engine.modules.tree_kda import Topology, conv, verify
from engine.modules.linear_attention import gated_delta_rule, kda_gate
from engine.modules.causal_conv import causal_conv1d


class TreePlanTests(unittest.TestCase):
    def test_invalid_topology_duplicate_siblings_and_ids(self):
        for tokens, parents in (((1, 2), (-1, 1)), ((1, 2), (-1, -1)), ((1, 2, 2), (-1, 0, 0)),
                               ((True,), (-1,)), ((-1,), (-1,)), ((1,), (False,)), ((), ())):
            with self.subTest(tokens=tokens, parents=parents), self.assertRaises(ValueError):
                Tree(tokens, parents)

    def test_greedy_bonus_eos_and_budget_commit_only_computed_inputs(self):
        tree = Tree((10, 20, 21, 30, 31), (-1, 0, 0, 1, 2))
        self.assertEqual(tree.greedy([21, 30, 31, 99, 7], budget=8), ((21, 31, 7), (0, 2, 4)))
        self.assertEqual(tree.greedy([21, 30, 31, 99, 7], budget=2), ((21, 31), (0, 2)))
        self.assertEqual(tree.greedy([21, 30, 31, 99, 7], budget=8, eos=frozenset({31})), ((21, 31), (0, 2)))
        self.assertEqual(tree.greedy([99, 30, 31, 99, 7], budget=8), ((99,), (0,)))

    def test_unique_experts_charged_per_layer_and_shared_children_win(self):
        candidates = (Candidate(1, -1, 1., frozenset({(1, 0)})),
                      Candidate(2, 0, .6, frozenset({(1, 1)})),
                      Candidate(3, 0, .4, frozenset({(1, 0)})),
                      Candidate(4, 2, .9, frozenset({(2, 0)})))
        chosen = select(candidates, 2, bytes_per_expert=100, fixed_node_bytes=1)
        self.assertEqual(chosen.source_nodes, (0, 2))
        self.assertEqual(chosen.predicted_expert_bytes, 100)
        self.assertEqual(select(candidates, 4, bytes_per_expert=100, fixed_node_bytes=1).predicted_expert_bytes, 300)
        self.assertEqual(select(candidates, 2, bytes_per_expert=100, fixed_node_bytes=1, cost_weight=0).source_nodes, (0, 1))

    def test_unknown_routes_not_free_and_probability_mass_not_acceptance(self):
        cs = (Candidate(1, -1, 1., frozenset({(0, 0)})), Candidate(2, 0, .6),
              Candidate(3, 0, .4, frozenset({(0, 0)})))
        chosen = select(cs, 2, bytes_per_expert=100, fixed_node_bytes=1)
        self.assertEqual(chosen.tree.tokens, (1, 3))
        self.assertAlmostEqual(chosen.proposal_mass, 1.4)
        plain = tuple(Candidate(c.token, c.parent, c.probability) for c in cs)
        self.assertEqual(select(plain, 2, bytes_per_expert=100, fixed_node_bytes=1).tree.tokens, (1, 2))

    def test_invalid_probabilities_and_costs(self):
        for prob in (float("nan"), -1., 1.1):
            with self.assertRaises(ValueError):
                select((Candidate(1, -1, prob),), 1, bytes_per_expert=1, fixed_node_bytes=1)
        with self.assertRaises(ValueError):
            select((Candidate(1, -1, 1.), Candidate(2, 0, .8), Candidate(3, 0, .8)), 2,
                   bytes_per_expert=1, fixed_node_bytes=1)

    def test_bounded_route_learning_and_runtime_identity(self):
        table = RouteTable("weights+selector-a", capacity=1)
        table.observe(1, 2, 1, {(3, 4)}, runtime_id=table.runtime_id)
        self.assertEqual(table.predict(1, 2, 1, runtime_id=table.runtime_id), frozenset())
        table.observe(1, 2, 1, {(3, 5)}, runtime_id=table.runtime_id)
        self.assertEqual(table.predict(1, 2, 1, runtime_id=table.runtime_id), frozenset({(3, 4), (3, 5)}))
        table.observe(2, 3, 2, {(3, 4)}, runtime_id=table.runtime_id)
        self.assertEqual(len(table.entries), 1)
        self.assertEqual(table.predict(1, 2, 1, runtime_id=table.runtime_id), frozenset())
        with self.assertRaises(ValueError):
            table.predict(1, 2, 1, runtime_id="other-weights")

    def test_dflash_edges_use_each_actual_parent_and_keep_pruned_mass(self):
        unary = torch.tensor([[0., 0.], [0., 0.]])
        tokens = torch.tensor([[2, 3], [4, 5]])
        pred = torch.arange(6).float()[:, None]
        succ = -pred
        rows = dflash_candidates(1, unary, tokens, torch.ones(2, 1), pred, succ, (1., 1.), width=2)
        self.assertEqual(len(rows), 7)
        for i, candidate in enumerate(rows[1:], 1):
            self.assertLess(candidate.parent, i)
            support = [2, 3] if candidate.parent == 0 else [4, 5]
            probability = torch.softmax(-torch.tensor(support).float() * rows[candidate.parent].token, 0)
            self.assertAlmostEqual(candidate.probability, float(probability[support.index(candidate.token)]))
        narrow = dflash_candidates(1, unary, tokens, torch.ones(2, 1), pred, succ, (1., 1.), width=1)
        self.assertLess(narrow[1].probability, 1.)

    def test_peaked_seven_step_draft_keeps_deep_path_with_same_node_budget(self):
        tokens = torch.arange(2, 16).view(7, 2)
        unary = torch.tensor([[5., 0.]]).expand(7, 2)
        codes = torch.zeros(16, 1)
        cs = dflash_candidates(1, unary, tokens, torch.zeros(7, 1), codes, codes, (1.,)*7,
                               width=2, max_nodes=8)
        tree = Tree(tuple(c.token for c in cs), tuple(c.parent for c in cs))
        self.assertEqual(tree.tokens, (1, 2, 4, 6, 8, 10, 12, 14))
        self.assertEqual(tree.depths, tuple(range(8)))
        self.assertEqual(tree.greedy((2, 4, 6, 8, 10, 12, 14, 99), budget=8)[0],
                         (2, 4, 6, 8, 10, 12, 14, 99))

    def test_compact_edges_preserve_ties_and_reject_duplicate_support(self):
        codes = torch.zeros(8, 1)
        args = (1, torch.zeros(2, 3), torch.tensor([[2, 3, 4], [5, 6, 7]]), torch.zeros(2, 1), codes, codes, (1., 1.))
        cs = dflash_candidates(*args, width=2, max_nodes=5)
        self.assertEqual([(c.token, c.parent) for c in cs], [(1, -1), (2, 0), (3, 0), (5, 1), (6, 1)])
        args[2][0, 1] = 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            dflash_candidates(*args)


class TreeStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(52)
        self.tree = Tree((1, 2, 3, 4, 5, 6), (-1, 0, 0, 1, 2, 4))

    def test_all_branches_match_independent_channelwise_kda_and_state(self):
        n, h, k, v = 6, 2, 7, 5
        q, key, raw = [torch.randn(n, h, k).bfloat16() for _ in range(3)]
        value, beta = torch.randn(n, h, v).bfloat16(), torch.randn(n, h).bfloat16()
        a, bias, initial = torch.randn(h), torch.randn(h*k), torch.randn(h, k, v)
        before = initial.clone()
        out, factors = verify(self.tree, q, key, value, raw, beta, a, bias, initial, -5.)
        for node in range(n):
            path = list(self.tree.path(node))
            expected, state = gated_delta_rule(q[path][None], key[path][None], value[path][None],
                kda_gate(raw[path][None], a, bias), beta[path][None].float().sigmoid(), initial[None],
                decay_per_channel=True)
            torch.testing.assert_close(out[node], expected[0, -1], atol=0, rtol=0)
            # torch's batched einsum and the factor oracle reduce differently;
            # require FP32 accuracy, without mistaking it for GPU bit equality.
            torch.testing.assert_close(factors.state(node), state[0], atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.equal(initial, before))
        self.assertEqual(factors.nbytes, 4*(h*k*v + n*h*(2*k+v)))
        self.assertEqual(factors.initial.dtype, torch.float32)

    def test_sibling_poison_does_not_enter_other_branch(self):
        q, k, v, g = [torch.randn(6, 2, 8).bfloat16() for _ in range(4)]
        b, a, bias, initial = torch.zeros(6, 2), torch.zeros(2), torch.zeros(16), torch.zeros(2, 8, 8)
        out, factors = verify(self.tree, q, k, v, g, b, a, bias, initial, -5.)
        v[1].fill_(float("nan"))
        changed, _ = verify(self.tree, q, k, v, g, b, a, bias, initial, -5.)
        torch.testing.assert_close(changed[[0, 2, 4, 5]], out[[0, 2, 4, 5]], atol=0, rtol=0)
        self.assertTrue(torch.isnan(changed[3]).all())

    def test_tree_conv_prefix_and_sibling_history(self):
        raw, weight, history = torch.randn(6, 9).bfloat16(), torch.randn(9, 4).bfloat16(), torch.randn(9, 3).bfloat16()
        out = conv(self.tree, raw, weight, history)
        for node in range(6):
            expected, _ = causal_conv1d(raw[list(self.tree.path(node))], weight, initial_state=history, activation="silu")
            torch.testing.assert_close(out[node], expected[-1], atol=0, rtol=0)

    def test_dfs_carries_chains_and_conv_uses_only_last_taps(self):
        self.assertEqual(self.tree.preorder, (0, 1, 3, 2, 4, 5))
        self.assertEqual(self.tree.state_updates, {"reconstruct": 15, "carry": 7})
        chain = Tree(tuple(range(8)), (-1, 0, 1, 2, 3, 4, 5, 6))
        self.assertEqual(chain.state_updates, {"reconstruct": 36, "carry": 8})
        topology = Topology(self.tree, "cpu")
        self.assertEqual(topology.conv.tolist(), [[-3, -2, -1, 0], [-2, -1, 0, 1], [-2, -1, 0, 2],
                                                [-1, 0, 1, 3], [-1, 0, 2, 4], [0, 2, 4, 5]])

    def test_reject_fp16_state(self):
        q = torch.zeros(6, 2, 8)
        with self.assertRaisesRegex(ValueError, "FP32"):
            verify(self.tree, q, q, q, q, torch.zeros(6, 2), torch.zeros(2), torch.zeros(16),
                   torch.zeros(2, 8, 8).half(), -5.)


if __name__ == "__main__":
    unittest.main()
