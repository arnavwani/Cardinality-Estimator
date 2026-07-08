import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from engine.catalog import Catalog
from engine.executor import true_cardinality, execute_plan
from estimators.histogram import HistogramEstimator
from estimators.learned import LearnedEstimator
from planner.join_optimizer import plan_query
from benchmark.generate_queries import random_query

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "moviedb.sqlite")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Compute q-error metric.
def q_error(true, pred):
    true, pred = max(true, 1), max(pred, 1)
    return max(true / pred, pred / true)

# Build train and test workloads.
def build_workload(catalog, n_train=1500, n_test_indist=300, n_test_heldout=150, seed=0):
    rng = np.random.RandomState(seed)
    # Sample query shape.
    def sample_shape():
        return rng.choice(["movies_only", "movies_casting", "full_3table"], p=[0.3, 0.3, 0.4])

    train_q = []
    while len(train_q) < n_train:
        shape = sample_shape()
        if shape == "full_3table":
            train_q.append(random_query(shape, n_movie_preds=rng.choice([0, 1])))
        else:
            train_q.append(random_query(shape))

    test_indist = []
    while len(test_indist) < n_test_indist:
        shape = sample_shape()
        if shape == "full_3table":
            test_indist.append(random_query(shape, n_movie_preds=rng.choice([0, 1])))
        else:
            test_indist.append(random_query(shape))

    test_heldout = [random_query("full_3table", n_movie_preds=2) for _ in range(n_test_heldout)]

    return train_q, test_indist, test_heldout

# Compute true cardinalities.
def label(catalog, queries):
    return [true_cardinality(catalog, q) for q in queries]

# Evaluate estimator accuracy.
def eval_estimator(estimator, queries, true_cards, label_str):
    errs = [q_error(t, estimator.estimate(q)) for t, q in zip(true_cards, queries)]
    errs = np.array(errs)
    print(f"  [{label_str:>10}] n={len(errs):4d}  median={np.median(errs):7.2f}  "
          f"p90={np.percentile(errs,90):7.2f}  max={errs.max():8.2f}")
    return errs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading catalog...")
    catalog = Catalog(DB_PATH)

    print("Generating workload...")
    train_q, test_indist, test_heldout = build_workload(catalog)

    print(f"Labeling {len(train_q)} train + {len(test_indist)} in-dist test + "
          f"{len(test_heldout)} held-out-shape test queries with true cardinalities...")
    train_true = label(catalog, train_q)
    indist_true = label(catalog, test_indist)
    heldout_true = label(catalog, test_heldout)

    print("Building histogram baseline...")
    hist = HistogramEstimator(catalog)

    print("Training learned estimator (XGBoost)...")
    learned = LearnedEstimator(hist)
    learned.fit(train_q, train_true)

    print("\n=== Estimation accuracy (q-error: max(true/pred, pred/true), lower is better) ===")
    print(" -- In-distribution test set --")
    h_in = eval_estimator(hist, test_indist, indist_true, "histogram")
    l_in = eval_estimator(learned, test_indist, indist_true, "learned")

    print(" -- Held-out query-shape test set (2 movie predicates, never seen in training) --")
    h_out = eval_estimator(hist, test_heldout, heldout_true, "histogram")
    l_out = eval_estimator(learned, test_heldout, heldout_true, "learned")

    print("\n=== End-to-end planner benchmark (multi-table in-distribution queries) ===")
    multi_table_q = [(q, t) for q, t in zip(test_indist, indist_true) if len(q.tables) > 1][:80]

    rows = []
    for q, true_card in multi_table_q:
        plan_h = plan_query(q, hist)
        plan_l = plan_query(q, learned)
        card_h, time_h = execute_plan(catalog, q, plan_h["join_order"], plan_h["algorithms"])
        card_l, time_l = execute_plan(catalog, q, plan_l["join_order"], plan_l["algorithms"])
        rows.append({
            "true_card": true_card,
            "hist_order": ">".join(plan_h["join_order"]), "hist_algos": ",".join(plan_h["algorithms"]),
            "hist_time_s": time_h, "hist_result_card": card_h,
            "learn_order": ">".join(plan_l["join_order"]), "learn_algos": ",".join(plan_l["algorithms"]),
            "learn_time_s": time_l, "learn_result_card": card_l,
        })

    df = pd.DataFrame(rows)
    assert (df["hist_result_card"] == df["true_card"]).all(), "engine correctness check failed (histogram plan)"
    assert (df["learn_result_card"] == df["true_card"]).all(), "engine correctness check failed (learned plan)"

    print(f"  queries benchmarked: {len(df)}")
    print(f"  total time, histogram-driven plans: {df['hist_time_s'].sum():.4f}s")
    print(f"  total time, learned-driven plans:   {df['learn_time_s'].sum():.4f}s")
    print(f"  mean per-query time, histogram: {df['hist_time_s'].mean()*1000:.3f}ms")
    print(f"  mean per-query time, learned:   {df['learn_time_s'].mean()*1000:.3f}ms")
    n_different_plan = (df["hist_order"] + df["hist_algos"] != df["learn_order"] + df["learn_algos"]).sum()
    print(f"  queries where the two estimators picked a DIFFERENT plan: {n_different_plan}/{len(df)}")

    df.to_csv(os.path.join(OUT_DIR, "planner_benchmark.csv"), index=False)

    summary = pd.DataFrame({
        "histogram_indist": h_in, "learned_indist": np.pad(l_in, (0, len(h_in) - len(l_in))) if len(l_in) < len(h_in) else l_in,
    })
    pd.DataFrame({"q_error": h_in, "estimator": "histogram", "split": "in_distribution"}).to_csv(
        os.path.join(OUT_DIR, "qerror_raw.csv"), index=False)
    for arr, est, split in [(h_in, "histogram", "in_distribution"), (l_in, "learned", "in_distribution"),
                             (h_out, "histogram", "held_out_shape"), (l_out, "learned", "held_out_shape")]:
        pd.DataFrame({"q_error": arr, "estimator": est, "split": split}).to_csv(
            os.path.join(OUT_DIR, f"qerror_{est}_{split}.csv"), index=False)

    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()