from dataclasses import dataclass, field
from typing import Any, List

# Represents a filter predicate.
@dataclass
class Predicate:
    table: str
    column: str
    op: str  # one of =, !=, <, <=, >, >=
    value: Any

# Represents a join condition.
@dataclass
class JoinCondition:
    left_table: str
    left_col: str
    right_table: str
    right_col: str

# Represents a parsed SQL query.
@dataclass
class Query:
    tables: List[str]
    joins: List[JoinCondition] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)

    def to_sql(self, select="COUNT(*)"):
        from_clause = self.tables[0]
        clauses = []
        for p in self.predicates:
            v = f"'{p.value}'" if isinstance(p.value, str) else p.value
            clauses.append(f"{p.table}.{p.column} {p.op} {v}")
        for j in self.joins:
            clauses.append(f"{j.left_table}.{j.left_col} = {j.right_table}.{j.right_col}")
        tables_sql = ", ".join(self.tables)
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        return f"SELECT {select} FROM {tables_sql} WHERE {where_sql}"