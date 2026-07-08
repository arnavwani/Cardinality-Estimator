import numpy as np
from engine.ir import Query, Predicate, JoinCondition

GENRES = ["Comedy", "Drama", "Action", "Horror", "Romance", "Documentary"]
NATIONALITIES = ["American", "Indian", "British", "French", "Japanese", "Korean"]

JOIN_M_C = JoinCondition("movies", "id", "casting", "movie_id")
JOIN_C_A = JoinCondition("casting", "actor_id", "actors", "id")

# Generate random year predicate.
def _rand_year_pred(table="movies"):
    op = np.random.choice(["=", ">", "<", ">=", "<="])
    val = int(np.random.uniform(1980, 2026))
    return Predicate(table, "year", op, val)

# Generate random gebre predicate.
def _rand_genre_pred(table="movies"):
    return Predicate(table, "genre", "=", np.random.choice(GENRES))

# Generate random rating predicate.
def _rand_rating_pred(table="movies"):
    op = np.random.choice([">", "<", ">=", "<="])
    val = round(float(np.random.uniform(1, 10)), 1)
    return Predicate(table, "rating", op, val)

# Generate random nationality predicate.
def _rand_nationality_pred(table="actors"):
    return Predicate(table, "nationality", "=", np.random.choice(NATIONALITIES))


MOVIE_PREDS = [_rand_year_pred, _rand_genre_pred, _rand_rating_pred]
ACTOR_PREDS = [_rand_nationality_pred]

# Sample random predicates.
def _sample_predicates(fns, table, k_max):
    k = np.random.randint(0, min(k_max, len(fns)) + 1)
    chosen = np.random.choice(len(fns), size=k, replace=False) if k > 0 else []
    return [fns[i](table) for i in chosen]

# Generate a random SQL query.
def random_query(shape=None, n_movie_preds=None):
    shape = shape or np.random.choice(["movies_only", "movies_casting", "full_3table"],
                                       p=[0.3, 0.3, 0.4])
    if shape == "movies_only":
        preds = _sample_predicates(MOVIE_PREDS, "movies", 3)
        while not preds:
            preds = _sample_predicates(MOVIE_PREDS, "movies", 3)
        return Query(tables=["movies"], joins=[], predicates=preds)

    if shape == "movies_casting":
        preds = _sample_predicates(MOVIE_PREDS, "movies", 2)
        return Query(tables=["movies", "casting"], joins=[JOIN_M_C], predicates=preds)

    if n_movie_preds is not None:
        idxs = np.random.choice(len(MOVIE_PREDS), size=min(n_movie_preds, len(MOVIE_PREDS)), replace=False)
        mpreds = [MOVIE_PREDS[i]("movies") for i in idxs]
    else:
        mpreds = _sample_predicates(MOVIE_PREDS, "movies", 2)
    apreds = _sample_predicates(ACTOR_PREDS, "actors", 1)
    preds = mpreds + apreds
    while not preds:
        apreds = _sample_predicates(ACTOR_PREDS, "actors", 1)
        preds = mpreds + apreds
    return Query(tables=["movies", "casting", "actors"], joins=[JOIN_M_C, JOIN_C_A], predicates=preds)

# Generate workload of random queries.
def generate_workload(n, shapes=None, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    queries = []
    for _ in range(n):
        shape = np.random.choice(shapes) if shapes else None
        queries.append(random_query(shape))
    np.random.set_state(rng_state)
    return queries