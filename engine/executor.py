import time
import pandas as pd
from .ir import Query


def _apply_predicates(df, table, predicates):
    for p in predicates:
        if p.table != table:
            continue
        col = df[p.column]
        if p.op == "=":
            df = df[col == p.value]
        elif p.op == "!=":
            df = df[col != p.value]
        elif p.op == "<":
            df = df[col < p.value]
        elif p.op == "<=":
            df = df[col <= p.value]
        elif p.op == ">":
            df = df[col > p.value]
        elif p.op == ">=":
            df = df[col >= p.value]
    return df

# Execute nested-loop join.
def nested_loop_join(left, right, left_col, right_col):
    left, right = _align_key_dtypes(left, right, left_col, right_col)
    common_cols = set(left.columns) & set(right.columns)
    rename_map = {c: (f"{c}_r" if c in common_cols else c) for c in right.columns}
    right_renamed = right.rename(columns=rename_map)
    right_col_name = rename_map[right_col]

    right_index = {}
    for _, rrow in right_renamed.iterrows():
        right_index.setdefault(rrow[right_col_name], []).append(rrow)

    out_rows = []
    for _, lrow in left.iterrows():
        for rrow in right_index.get(lrow[left_col], []):
            out_rows.append({**lrow.to_dict(), **rrow.to_dict()})

    if not out_rows:
        return left.iloc[0:0].merge(right_renamed.iloc[0:0], left_on=left_col, right_on=right_col_name)
    return pd.DataFrame(out_rows)

# Align join key data types.
def _align_key_dtypes(left, right, left_col, right_col):
    if left[left_col].dtype != right[right_col].dtype:
        target = left[left_col].dtype if len(left) > 0 else right[right_col].dtype
        try:
            left[left_col] = left[left_col].astype(target)
            right[right_col] = right[right_col].astype(target)
        except (ValueError, TypeError):
            pass
    return left, right

# Execute hash join.
def hash_join(left, right, left_col, right_col):
    left, right = _align_key_dtypes(left, right, left_col, right_col)
    # pandas.merge is a real hash join under the hood -- use it directly
    return left.merge(right, left_on=left_col, right_on=right_col, suffixes=("", "_r"))


ALGORITHMS = {"nested_loop": nested_loop_join, "hash": hash_join}

# Find the next valid join edge.
def find_join_edge(joins, acc_tables, next_table):
    for j in joins:
        if j.left_table in acc_tables and j.right_table == next_table:
            return j.left_table, j.left_col, next_table, j.right_col
        if j.right_table in acc_tables and j.left_table == next_table:
            return j.right_table, j.right_col, next_table, j.left_col
    raise ValueError(f"No join edge found connecting {next_table} to {acc_tables}")

# Execute the selected query plan.
def execute_plan(catalog, query: Query, join_order, algorithms):
    t0 = time.perf_counter()
    first = join_order[0]
    acc = _apply_predicates(catalog.tables[first].copy(), first, query.predicates)
    acc_tables = {first}

    for step, next_table in enumerate(join_order[1:]):
        right_df = _apply_predicates(catalog.tables[next_table].copy(), next_table, query.predicates)
        lt, lcol, rt, rcol = find_join_edge(query.joins, acc_tables, next_table)
        left_col_name = lcol if lcol in acc.columns else f"r_{lcol}"
        algo_fn = ALGORITHMS[algorithms[step]]
        acc = algo_fn(acc, right_df, left_col_name, rcol)
        acc_tables.add(next_table)

    elapsed = time.perf_counter() - t0
    return len(acc), elapsed

# Compute exact result cardinality.
def true_cardinality(catalog, query: Query) -> int:
    sql = query.to_sql(select="COUNT(*)")
    return catalog.true_cardinality(sql)