import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.parser import parse_sql
from engine.catalog import Catalog
from engine.executor import true_cardinality
from estimators.histogram import HistogramEstimator

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "moviedb.sqlite")


def test_histogram_single_predicate_reasonable():
    catalog = Catalog(DB_PATH)
    hist = HistogramEstimator(catalog)
    q = parse_sql("SELECT * FROM movies WHERE movies.genre = 'Comedy'")
    true_card = true_cardinality(catalog, q)
    est = hist.estimate(q)
    assert 0.5 * true_card <= est <= 1.5 * true_card


def test_histogram_underestimates_on_known_correlation():
    catalog = Catalog(DB_PATH)
    hist = HistogramEstimator(catalog)
    q = parse_sql("SELECT * FROM movies WHERE movies.genre = 'Comedy' AND movies.year > 2015")
    true_card = true_cardinality(catalog, q)
    est = hist.estimate(q)
    q_err = max(true_card / max(est, 1), est / max(true_card, 1))
    # meaningfully off, demonstrating the independence-assumption flaw
    assert q_err > 1.3