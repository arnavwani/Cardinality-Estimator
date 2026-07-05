
import numpy as np
import pandas as pd
from .base import Estimator

# Histogram for a single column.
class Histogram:
    def __init__(self, values: pd.Series, n_buckets=20):
        self.is_numeric = pd.api.types.is_numeric_dtype(values)
        self.n = len(values)
        if self.is_numeric:
            quantiles = np.linspace(0, 1, n_buckets + 1)
            self.edges = np.unique(values.quantile(quantiles).values)
            if len(self.edges) < 2:
                self.edges = np.array([values.min(), values.max() + 1])
            counts, _ = np.histogram(values, bins=self.edges)
            self.counts = counts
        else:
            vc = values.value_counts()
            self.freq = (vc / self.n).to_dict()
            self.distinct = len(vc)

    def selectivity(self, op, value):
        if self.is_numeric:
            return self._numeric_selectivity(op, value)
        return self._categorical_selectivity(op, value)

    def _numeric_selectivity(self, op, value):
        edges, counts = self.edges, self.counts
        total = counts.sum()
        if total == 0:
            return 0.0
        if op == "=":
            idx = np.searchsorted(edges, value, side="right") - 1
            idx = min(max(idx, 0), len(counts) - 1)
            bucket_width = max(edges[idx + 1] - edges[idx], 1e-9)
            return (counts[idx] / bucket_width) / total
        if op in (">", ">="):
            idx = np.searchsorted(edges, value, side="left")
            frac_in_boundary_bucket = 0.0
            if 0 < idx <= len(counts):
                b = idx - 1
                if b < len(counts):
                    bucket_lo, bucket_hi = edges[b], edges[b + 1]
                    width = max(bucket_hi - bucket_lo, 1e-9)
                    frac_in_boundary_bucket = max(0.0, (bucket_hi - value) / width) * counts[b]
            rest = counts[idx:].sum() if idx < len(counts) else 0
            return (rest + frac_in_boundary_bucket) / total
        if op in ("<", "<="):
            return 1 - self._numeric_selectivity(">=" if op == "<" else ">", value)
        if op == "!=":
            return 1 - self._numeric_selectivity("=", value)
        return 0.5

    def _categorical_selectivity(self, op, value):
        p = self.freq.get(value, 1.0 / max(self.distinct, 1) * 0.1)
        if op == "=":
            return p
        if op == "!=":
            return 1 - p
        return 0.5


class HistogramEstimator(Estimator):
    name = "histogram"

    def __init__(self, catalog):
        self.catalog = catalog
        self.histograms = {}
        for table, df in catalog.tables.items():
            for col in df.columns:
                self.histograms[(table, col)] = Histogram(df[col])

    def estimate(self, query) -> float:
        table_est = {}
        for t in query.tables:
            sel = 1.0
            for p in query.predicates:
                if p.table == t:
                    sel *= self.histograms[(t, p.column)].selectivity(p.op, p.value)
            table_est[t] = max(self.catalog.row_count(t) * sel, 0.0)

        if not query.joins:
            # single table
            return table_est[query.tables[0]]

        acc_tables = {query.tables[0]}
        acc_card = table_est[query.tables[0]]
        remaining_joins = list(query.joins)

        while remaining_joins:
            progressed = False
            for j in list(remaining_joins):
                if j.left_table in acc_tables and j.right_table not in acc_tables:
                    nxt, ncol, dcol, dtable = j.right_table, j.right_col, j.left_col, j.left_table
                elif j.right_table in acc_tables and j.left_table not in acc_tables:
                    nxt, ncol, dcol, dtable = j.left_table, j.left_col, j.right_col, j.right_table
                else:
                    continue
                d_next = self.catalog.distinct_count(nxt, ncol)
                d_acc = self.catalog.distinct_count(dtable, dcol)
                denom = max(d_next, d_acc, 1)
                acc_card = (acc_card * table_est[nxt]) / denom
                acc_tables.add(nxt)
                remaining_joins.remove(j)
                progressed = True
            if not progressed:
                break  # disconnected join graph, shouldn't happen with our query generator

        return max(acc_card, 0.0)