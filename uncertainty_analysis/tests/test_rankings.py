import itertools
import numpy as np
import pytest
from uncertainty_analysis.common import pairwise_counts, order_scores, rank_metrics
from uncertainty_analysis.point_2_adjusted_scores import select_lambda
from uncertainty_analysis.point_3_rank_distributions import rank_distribution
from uncertainty_analysis.consensus_solver import agreement, consensus, ordered_components, violated_triangles


def brute_force(w):
    return max(agreement(w, p) for p in itertools.permutations(range(len(w))))


def test_pairwise_uses_matched_heads_and_preserves_ties():
    p = np.array([[.1, .9], [.2, 1.], [.1, .9]])
    wins, ties = pairwise_counts(p)
    assert wins[0, 1] == 0  # Cross-head comparisons would incorrectly give positive mass.
    assert wins[1, 0] == 2
    assert ties[0, 2] == 2
    assert np.all(wins + wins.T + ties == 2)
    assert np.all(wins.diagonal() == 0)


def test_midranks_and_fractional_boundary_ties():
    p = np.array([[.9, .9], [.9, .9], [.1, .1]])
    ranks, summary = rank_distribution(p, 1)
    assert np.allclose(ranks[:, 0], [1.5, 1.5, 3])
    assert np.allclose(summary['p_top_k'], [.5, .5, 0])
    assert np.allclose(summary['p_first'], [.5, .5, 0])
    assert np.allclose(summary['rank_q025'], summary['rank_q975'])


def test_expected_rank_matches_pairwise_borda():
    rng = np.random.default_rng(45)
    p = rng.integers(0, 5, (12, 128)) / 5
    _, summary = rank_distribution(p, 4)
    wins, ties = pairwise_counts(p)
    expected = 1 + (wins.T + .5 * ties).sum(1) / 128 - .5
    assert np.allclose(summary['expected_rank'], expected)
    assert np.isclose(summary['p_top_k'].sum(), 4)
    assert np.isclose(summary['p_first'].sum(), 1)


def test_lambda_fits_validation_and_prefers_smallest_tied_value():
    p = np.array([[1., .6], [.72, .72], [.1, .1]])
    # Unstable false hazard wins at zero; stable true hazard wins with a positive penalty.
    y = np.array([0, 1, 0]); ids = np.array(['a', 'b', 'c'])
    value, sweep = select_lambda(p, y, ids, 1, grid=[0, .5, 1])
    assert value == .5
    assert sweep.hits.tolist() == [0, 1, 1]
    stable = np.array([[.9, .9], [.7, .7], [.1, .1]])
    assert select_lambda(stable, y, ids, 1, grid=[0, .5, 1])[0] == 0


def test_ranking_ids_and_precision_are_deterministic():
    ids = np.array(['z', 'a', 'b'])
    order = order_scores([.7, .7, .2], ids)
    assert order.tolist() == [1, 0, 2]
    result = rank_metrics([0, 1, 1], order, 2)
    assert result['hits'] == 1 and result['precision_at_k'] == .5


@pytest.mark.parametrize('seed', range(6))
def test_kemeny_matches_exhaustive_optimum(seed):
    rng = np.random.default_rng(seed)
    p = rng.integers(0, 6, (6, 13)) / 6
    wins, ties = pairwise_counts(p)
    expected = (wins.T + .5 * ties).sum(1)
    initial = order_scores(expected, np.array(list('abcdef')), descending=False)
    order, info = consensus(wins, np.array(list('abcdef')), initial, seconds=10)
    optimum = brute_force(wins)
    assert agreement(wins, order) == optimum
    assert info['optimal'] and info['upper_bound'] == optimum
    assert sorted(order) == list(range(6))


def test_cyclic_majorities_and_scc_decomposition():
    # Three rankings: ABC, BCA, CAB create A>B, B>C, C>A; D always last.
    p = np.array([[.9, .1, .5], [.5, .9, .1], [.1, .5, .9], [0, 0, 0]])
    wins, _ = pairwise_counts(p)
    assert [len(g) for g in ordered_components(wins)] == [3, 1]
    assert violated_triangles(wins > wins.T) == [(0, 1, 2)]
    order, info = consensus(wins, list('abcd'), np.arange(4), seconds=10)
    assert info['optimal'] and agreement(wins, order) == brute_force(wins)
    assert order[-1] == 3


def test_timeout_retains_feasible_ranking_and_valid_bound():
    rng = np.random.default_rng(29)
    wins, _ = pairwise_counts(rng.random((7, 9)))
    initial = np.arange(7)
    order, info = consensus(wins, list('abcdefg'), initial, seconds=0)
    optimum = brute_force(wins)
    assert agreement(wins, order) >= agreement(wins, initial)
    assert agreement(wins, order) <= optimum <= info['upper_bound'] + 1e-7


def test_canonical_ids_make_consensus_row_order_invariant():
    rng = np.random.default_rng(32)
    wins, ties = pairwise_counts(rng.integers(0, 4, (7, 15))/4)
    ids = np.array(list('gfedcba'))
    initial = order_scores((wins.T + .5*ties).sum(1), ids, descending=False)
    order, _ = consensus(wins, ids, initial, seconds=10)
    perm = rng.permutation(7)
    reordered = wins[np.ix_(perm,perm)]
    reordered_ties = ties[np.ix_(perm,perm)]
    initial2 = order_scores((reordered.T+.5*reordered_ties).sum(1),ids[perm],descending=False)
    order2, _ = consensus(reordered,ids[perm],initial2,seconds=10)
    assert np.array_equal(ids[order],ids[perm][order2])
