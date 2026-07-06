import itertools
from engine.ir import Query
from .cost_model import best_algo_and_cost

# Build a subquery for estimation.
def _subquery(query: Query, table_subset):
    subset = set(table_subset)
    joins = [j for j in query.joins if j.left_table in subset and j.right_table in subset]
    predicates = [p for p in query.predicates if p.table in subset]
    return Query(tables=list(table_subset), joins=joins, predicates=predicates)

# Validate join order connectivity.
def _is_connected_prefix(query: Query, prefix):
    acc = {prefix[0]}
    for t in prefix[1:]:
        connected = any(
            (j.left_table in acc and j.right_table == t) or (j.right_table in acc and j.left_table == t)
            for j in query.joins
        )
        if not connected:
            return False
        acc.add(t)
    return True

# Find the lowest-cost execution plan.
def plan_query(query: Query, estimator, verbose=False):
    if len(query.tables) == 1:
        return {"join_order": query.tables, "algorithms": [], "estimated_cost": 0.0, "step_cards": []}

    best = None
    for perm in itertools.permutations(query.tables):
        if not _is_connected_prefix(query, perm):
            continue
        total_cost = 0.0
        algos = []
        step_cards = []
        acc_tables = [perm[0]]
        acc_card = estimator.estimate(_subquery(query, acc_tables))
        for next_table in perm[1:]:
            right_card = estimator.estimate(_subquery(query, [next_table]))
            algo, cost = best_algo_and_cost(acc_card, right_card)
            algos.append(algo)
            total_cost += cost
            acc_tables.append(next_table)
            acc_card = estimator.estimate(_subquery(query, acc_tables))
            step_cards.append(acc_card)
        candidate = {"join_order": list(perm), "algorithms": algos,
                     "estimated_cost": total_cost, "step_cards": step_cards}
        if verbose:
            print(candidate)
        if best is None or candidate["estimated_cost"] < best["estimated_cost"]:
            best = candidate
    return best