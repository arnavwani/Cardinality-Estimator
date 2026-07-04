import re
from .ir import Query, Predicate, JoinCondition

OPS = ["!=", ">=", "<=", "=", ">", "<"]

# Convert SQL literal to Python value.
def _parse_value(raw):
    raw = raw.strip()
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw

# Parse a single WHERE/JOIN condition.
def _parse_predicate(clause):
    clause = clause.strip()
    for op in OPS:
        if op in clause:
            left, right = clause.split(op, 1)
            table, col = left.strip().split(".")
            return Predicate(table=table.strip(), column=col.strip(),
                              op=op, value=_parse_value(right))
    raise ValueError(f"Could not parse predicate: {clause}")

# Parse SQL into Query IR.
def parse_sql(sql: str) -> Query:
    sql = " ".join(sql.strip().rstrip(";").split())

    where_split = re.split(r"\bWHERE\b", sql, flags=re.IGNORECASE)
    head = where_split[0]
    where_clause = where_split[1] if len(where_split) > 1 else ""

    head = re.sub(r"^SELECT\s+\*\s+FROM\s+", "", head, flags=re.IGNORECASE).strip()

    join_parts = re.split(r"\bJOIN\b", head, flags=re.IGNORECASE)
    base_table = join_parts[0].strip()
    tables = [base_table]
    joins = []
    for part in join_parts[1:]:
        m = re.match(r"\s*(\w+)\s+ON\s+(.+)", part, flags=re.IGNORECASE)
        table_name = m.group(1).strip()
        on_clause = m.group(2).strip()
        left, right = on_clause.split("=")
        lt, lc = left.strip().split(".")
        rt, rc = right.strip().split(".")
        tables.append(table_name)
        joins.append(JoinCondition(lt.strip(), lc.strip(), rt.strip(), rc.strip()))

    predicates = []
    if where_clause.strip():
        parts = re.split(r"\bAND\b", where_clause, flags=re.IGNORECASE)
        predicates = [_parse_predicate(p) for p in parts if p.strip()]

    return Query(tables=tables, joins=joins, predicates=predicates)