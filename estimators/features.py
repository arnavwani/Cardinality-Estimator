import numpy as np

GENRES = ["Comedy", "Drama", "Action", "Horror", "Romance", "Documentary"]
NATIONALITIES = ["American", "Indian", "British", "French", "Japanese", "Korean"]
OPS = ["=", "!=", "<", "<=", ">", ">="]

YEAR_MIN, YEAR_MAX = 1980, 2026
RATING_MIN, RATING_MAX = 1.0, 10.0

FEATURE_NAMES = (
    ["has_casting", "has_actors"]
    + ["genre_present"] + [f"genre_{g}" for g in GENRES]
    + ["year_present"] + [f"year_op_{o}" for o in OPS] + ["year_value_norm"]
    + ["rating_present"] + [f"rating_op_{o}" for o in OPS] + ["rating_value_norm"]
    + ["nat_present"] + [f"nat_{n}" for n in NATIONALITIES]
    + ["join_movies_casting", "join_casting_actors"]
    + ["hist_log_estimate"]
)

# One-hot encode a value.
def _onehot(value, options):
    return [1.0 if value == o else 0.0 for o in options]

# Convert query into features.
def featurize(query, hist_estimator) -> np.ndarray:
    tables = set(query.tables)
    has_casting = 1.0 if "casting" in tables else 0.0
    has_actors = 1.0 if "actors" in tables else 0.0

    genre_pred = next((p for p in query.predicates if p.table == "movies" and p.column == "genre"), None)
    year_pred = next((p for p in query.predicates if p.table == "movies" and p.column == "year"), None)
    rating_pred = next((p for p in query.predicates if p.table == "movies" and p.column == "rating"), None)
    nat_pred = next((p for p in query.predicates if p.table == "actors" and p.column == "nationality"), None)

    feats = [has_casting, has_actors]

    feats += [1.0 if genre_pred else 0.0] + _onehot(genre_pred.value if genre_pred else None, GENRES)

    year_val_norm = 0.0
    if year_pred:
        year_val_norm = (year_pred.value - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    feats += [1.0 if year_pred else 0.0] + _onehot(year_pred.op if year_pred else None, OPS) + [year_val_norm]

    rating_val_norm = 0.0
    if rating_pred:
        rating_val_norm = (rating_pred.value - RATING_MIN) / (RATING_MAX - RATING_MIN)
    feats += [1.0 if rating_pred else 0.0] + _onehot(rating_pred.op if rating_pred else None, OPS) + [rating_val_norm]

    feats += [1.0 if nat_pred else 0.0] + _onehot(nat_pred.value if nat_pred else None, NATIONALITIES)

    join_names = {(j.left_table, j.right_table) for j in query.joins} | {(j.right_table, j.left_table) for j in query.joins}
    feats += [1.0 if ("movies", "casting") in join_names else 0.0,
              1.0 if ("casting", "actors") in join_names else 0.0]

    # residual-learning signal: let the model start from the histogram estimate
    hist_est = hist_estimator.estimate(query)
    feats += [np.log1p(max(hist_est, 0.0))]

    return np.array(feats, dtype=np.float32)

# Convert multiple queries to features.
def featurize_batch(queries, hist_estimator) -> np.ndarray:
    return np.stack([featurize(q, hist_estimator) for q in queries])
