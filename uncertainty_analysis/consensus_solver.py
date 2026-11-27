"""Kemeny pairwise consensus with SCC decomposition and bounded exact search.

Maximize sum of within-head strict win counts in the forward direction of a
permutation. Half-credit for score ties changes this objective by a constant.
No true labels enter any solver routine.
"""
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import connected_components


def agreement(wins, order):
    return int(np.triu(wins[np.ix_(order, order)], 1).sum())


def insertion_search(wins, order, max_moves=2000):
    """Deterministic best positive-gain insertion; never reduces agreement."""
    order = np.array(order, dtype=int)
    for _ in range(max_moves):
        margin = wins[np.ix_(order, order)] - wins[np.ix_(order, order)].T
        best_gain, move = 0, None
        for i in range(len(order)):
            if i:
                gains = np.cumsum(margin[i, :i][::-1])[::-1]
                j = int(np.argmax(gains))
                if gains[j] > best_gain:
                    best_gain, move = int(gains[j]), (i, j)
            if i + 1 < len(order):
                gains = np.cumsum(-margin[i, i + 1:])
                offset = int(np.argmax(gains))
                if gains[offset] > best_gain:
                    best_gain, move = int(gains[offset]), (i, i + 1 + offset)
        if move is None:
            break
        i, j = move
        item = order[i]
        order = np.insert(np.delete(order, i), j, item)
    return order


def ordered_components(wins):
    """Tie edges go both ways; all cross-SCC preferences are strict/consistent."""
    adjacency = wins >= wins.T
    np.fill_diagonal(adjacency, False)
    count, labels = connected_components(csr_matrix(adjacency), directed=True, connection='strong')
    groups = [np.flatnonzero(labels == c) for c in range(count)]
    edges = np.zeros((count, count), dtype=bool)
    i, j = np.where(adjacency)
    edges[labels[i], labels[j]] = True
    np.fill_diagonal(edges, False)
    indegree = edges.sum(0)
    remaining, output = set(range(count)), []
    while remaining:
        available = [c for c in remaining if indegree[c] == 0]
        assert available, 'Condensation graph must be acyclic.'
        c = min(available, key=lambda x: int(groups[x].min()))
        output.append(groups[c]); remaining.remove(c)
        indegree -= edges[c]
    return output


def violated_triangles(orientation):
    """A complete directed order is transitive iff it has no directed triangle."""
    n = len(orientation)
    cuts = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            a = orientation[i, j]
            ks = np.flatnonzero((orientation[j, j + 1:] == a) &
                                (orientation[i, j + 1:] != a)) + j + 1
            cuts.extend((i, j, int(k)) for k in ks)
    return cuts


def exact_component(wins, initial, seconds):
    """Triangle-cut MILP, returning a feasible permutation and a valid bound."""
    n = len(wins)
    start = time.monotonic()
    best = insertion_search(wins, initial)
    best_score = agreement(wins, best)
    upper = float(np.maximum(wins, wins.T)[np.triu_indices(n, 1)].sum())
    info = dict(n=n, initial_agreement=agreement(wins, initial), heuristic_agreement=best_score,
                rounds=0, triangle_constraints=0, solver_status='pairwise_bound')
    if upper - best_score < 1e-7 or n <= 2:
        info.update(agreement=best_score, upper_bound=upper, optimal=True, elapsed_seconds=time.monotonic()-start)
        return best, info
    ii, jj = np.triu_indices(n, 1)
    variables = len(ii)
    indices = np.full((n, n), -1, dtype=int)
    indices[ii, jj] = np.arange(variables)
    c = -(wins[ii, jj] - wins[jj, ii]).astype(float)
    base = int(wins[jj, ii].sum())
    position = np.argsort(best)
    orientation = wins > wins.T
    orientation[wins == wins.T] = (position[:, None] < position[None, :])[wins == wins.T]
    np.fill_diagonal(orientation, False)
    cuts = set(violated_triangles(orientation))
    all_rounds = []
    while seconds > 0 and time.monotonic() - start < seconds:
        ordered_cuts = sorted(cuts)
        if ordered_cuts:
            triple = np.asarray(ordered_cuts)
            columns = np.column_stack([indices[triple[:, 0], triple[:, 1]],
                                       indices[triple[:, 1], triple[:, 2]],
                                       indices[triple[:, 0], triple[:, 2]]]).ravel()
            a = coo_matrix((np.tile([1., 1., -1.], len(triple)),
                            (np.repeat(np.arange(len(triple)), 3), columns)),
                           shape=(len(triple), variables)).tocsc()
        else:
            a = coo_matrix((0, variables)).tocsc()
        # A known feasible ordering supplies a safe objective cut (not a warm start).
        a = vstack([a, csr_matrix(c.reshape(1, -1))], format='csc')
        lower = np.r_[np.zeros(len(ordered_cuts)), -np.inf]
        higher = np.r_[np.ones(len(ordered_cuts)), base - best_score + 1e-7]
        remaining = seconds - (time.monotonic() - start)
        if remaining <= 0:
            break
        result = milp(c, integrality=np.ones(variables), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(a, lower, higher),
                      options=dict(time_limit=max(.01, remaining), mip_rel_gap=0., presolve=True))
        info['rounds'] += 1
        info['solver_status'] = str(result.message)
        dual = getattr(result, 'mip_dual_bound', None)
        if dual is not None and np.isfinite(dual):
            upper = min(upper, base - float(dual))
        assert upper + 1e-4 >= best_score, 'Invalid solver bound.'
        entry = dict(status=int(result.status), constraints=len(cuts), dual_bound=dual)
        all_rounds.append(entry)
        if result.x is None:
            break
        x = np.rint(result.x).astype(bool)
        if np.max(np.abs(result.x - x)) > 1e-5:
            break
        orientation = np.zeros((n, n), dtype=bool)
        orientation[ii, jj] = x; orientation[jj, ii] = ~x
        violated = set(violated_triangles(orientation))
        entry['violated_triangles'] = len(violated)
        if not violated:
            candidate = np.argsort(-orientation.sum(1), kind='stable')
            assert np.all(orientation[np.ix_(candidate, candidate)][ii, jj])
            score = agreement(wins, candidate)
            if score > best_score or (score == best_score and tuple(candidate) < tuple(best)):
                best, best_score = candidate, score
            break
        # A cyclic incumbent is not a ranking. Repair a degree order for a primal bound.
        candidate = insertion_search(wins, np.argsort(-orientation.sum(1), kind='stable'))
        score = agreement(wins, candidate)
        if score > best_score:
            best, best_score = candidate, score
        if upper - best_score < 1 - 1e-6:
            break  # Objective counts are integers, so no better feasible score exists.
        new = violated - cuts
        if not new:
            break
        cuts.update(new)
    certified = upper - best_score < 1 - 1e-6
    if certified:
        upper = float(best_score)
    info.update(agreement=best_score, upper_bound=upper, optimal=bool(certified),
                elapsed_seconds=time.monotonic()-start, triangle_constraints=len(cuts), solver_rounds=all_rounds)
    return best, info


def consensus(wins, ids, initial, seconds=240):
    """Exact SCC decomposition; MILP budget shared across nontrivial components."""
    wins = np.asarray(wins, dtype=np.int64)
    assert wins.shape == (len(ids), len(ids)) and (wins >= 0).all()
    assert np.all(np.diag(wins) == 0)
    # Canonicalize to sample IDs, so arbitrary input row order cannot change the heuristic.
    canonical = np.argsort(np.asarray(ids, dtype=str), kind='stable')
    w = wins[np.ix_(canonical, canonical)]
    inverse = np.argsort(canonical)
    original_rank = np.argsort(inverse[np.asarray(initial)])
    groups = ordered_components(w)
    start = time.monotonic()
    solved, info = [], []
    # Solve larger components first, with all components retaining a feasible fallback.
    component_results = {}
    for component_index in sorted(range(len(groups)), key=lambda i: -len(groups[i])):
        g = groups[component_index]
        local_w = w[np.ix_(g, g)]
        local_initial = np.argsort(original_rank[g], kind='stable')
        order, record = exact_component(local_w, local_initial, max(0, seconds-(time.monotonic()-start)))
        record['component_index'] = component_index
        component_results[component_index] = (g[order], record)
    for i in range(len(groups)):
        order, record = component_results[i]; solved.extend(order); info.append(record)
    output = canonical[np.asarray(solved)]
    score = agreement(wins, output)
    internal_score = sum(x['agreement'] for x in info)
    cross_score = score - internal_score
    upper = cross_score + sum(x['upper_bound'] for x in info)
    assert score >= agreement(wins, initial), 'Consensus must retain or improve initial agreement.'
    return output, dict(agreement=score, upper_bound=upper, absolute_gap=upper-score,
                        optimal=all(x['optimal'] for x in info), elapsed_seconds=time.monotonic()-start,
                        n_components=len(groups), component_sizes=[len(g) for g in groups],
                        components=info, initial_agreement=agreement(wins, initial))
