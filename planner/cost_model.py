
# Estimate cost of one join step.
def step_cost(algo, left_card, right_card):
    if algo == "nested_loop":
        return left_card + left_card * right_card
    if algo == "hash":
        return left_card + right_card
    raise ValueError(algo)

# Select the cheapest join algorithm.
def best_algo_and_cost(left_card, right_card):
    costs = {a: step_cost(a, left_card, right_card) for a in ("nested_loop", "hash")}
    best = min(costs, key=costs.get)
    return best, costs[best]
