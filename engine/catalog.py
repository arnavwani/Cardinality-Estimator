import sqlite3
import pandas as pd

#Loads tables into pandas DataFrames and keeps basic stats (row counts, distinct counts) used by the histogram estimator and the cost model.
class Catalog:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.tables = {}
        self._load_all()

    def _load_all(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = [r[0] for r in cur.fetchall()]
        for name in names:
            self.tables[name] = pd.read_sql_query(f"SELECT * FROM {name}", self.conn)

    def row_count(self, table):
        return len(self.tables[table])

    def distinct_count(self, table, column):
        return self.tables[table][column].nunique()

    def column_values(self, table, column):
        return self.tables[table][column]

    def true_cardinality(self, sql_count_query: str) -> int:
        cur = self.conn.cursor()
        cur.execute(sql_count_query)
        return cur.fetchone()[0]