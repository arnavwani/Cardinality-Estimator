import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Load benchmark results.
def load(est, split):
    return pd.read_csv(os.path.join(RESULTS_DIR, f"qerror_{est}_{split}.csv"))["q_error"].values

# Generate evaluation plots.
def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, split, title in zip(axes, ["in_distribution", "held_out_shape"],
                                 ["In-distribution test queries", "Held-out query-shape test queries"]):
        h = load("histogram", split)
        l = load("learned", split)
        bins = np.logspace(0, np.log10(max(h.max(), l.max(), 10)), 30)
        ax.hist(h, bins=bins, alpha=0.6, label=f"histogram (median={np.median(h):.2f})")
        ax.hist(l, bins=bins, alpha=0.6, label=f"learned (median={np.median(l):.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("q-error (log scale)")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "qerror_distributions.png")
    plt.savefig(out_path, dpi=130)
    print("Saved", out_path)

    # planner timing comparison
    df = pd.read_csv(os.path.join(RESULTS_DIR, "planner_benchmark.csv"))
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.bar(["histogram-driven plans", "learned-driven plans"],
            [df["hist_time_s"].sum(), df["learn_time_s"].sum()], color=["#d95f5f", "#5f9ed9"])
    ax2.set_ylabel("total execution time (s), 80 queries")
    ax2.set_title("End-to-end planner runtime")
    plt.tight_layout()
    out_path2 = os.path.join(RESULTS_DIR, "planner_runtime.png")
    plt.savefig(out_path2, dpi=130)
    print("Saved", out_path2)


if __name__ == "__main__":
    main()