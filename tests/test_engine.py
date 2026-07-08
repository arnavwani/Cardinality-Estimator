import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.parser import parse_sql
from engine.catalog import Catalog
from engine.executor import true_cardinality, execute_plan

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "moviedb.sqlite")


def test_parser_basic():
    q = parse_sql("SELECT * FROM movies WHERE movies.genre = 'Comedy' AND movies.year > 2015")
    assert q.tables == ["movies"]
    assert len(q.predicates) == 2
    assert q.predicates[0].op == "="
    assert q.predicates[1].op == ">"


def test_parser_joins():
    q = parse_sql(
        "SELECT * FROM movies JOIN casting ON movies.id = casting.movie_id "
        "JOIN actors ON casting.actor_id = actors.id WHERE actors.nationality = 'Indian'"
    )
    assert q.tables == ["movies", "casting", "actors"]
    assert len(q.joins) == 2


def test_engine_matches_sqlite_ground_truth():
    catalog = Catalog(DB_PATH)
    q = parse_sql(
        "SELECT * FROM movies JOIN casting ON movies.id = casting.movie_id "
        "JOIN actors ON casting.actor_id = actors.id "
        "WHERE movies.genre = 'Comedy' AND actors.nationality = 'Indian'"
    )
    true_card = true_cardinality(catalog, q)

    card_hash, _ = execute_plan(catalog, q, ["movies", "casting", "actors"], ["hash", "hash"])
    card_nl, _ = execute_plan(catalog, q, ["movies", "casting", "actors"], ["nested_loop", "nested_loop"])
    card_reordered, _ = execute_plan(catalog, q, ["actors", "casting", "movies"], ["hash", "hash"])

    assert card_hash == true_card
    assert card_nl == true_card
    assert card_reordered == true_card
    assert true_card > 0  # sanity: this correlated slice is non-empty


def test_single_table_no_join():
    catalog = Catalog(DB_PATH)
    q = parse_sql("SELECT * FROM movies WHERE movies.genre = 'Comedy'")
    true_card = true_cardinality(catalog, q)
    card, _ = execute_plan(catalog, q, ["movies"], [])
    assert card == true_card
