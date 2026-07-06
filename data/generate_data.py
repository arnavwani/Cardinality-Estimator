import sqlite3
import numpy as np
import os

np.random.seed(42)

OUT_DB = os.path.join(os.path.dirname(__file__), "..", "moviedb.sqlite")

GENRES = ["Comedy", "Drama", "Action", "Horror", "Romance", "Documentary"]
NATIONALITIES = ["American", "Indian", "British", "French", "Japanese", "Korean"]

N_MOVIES = 6000
N_ACTORS = 3000
N_CAST = 18000

# Generate synthetic movie records.
def gen_movies(n):
    ids, titles, genres, years, ratings = [], [], [], [], []
    for i in range(n):
        genre = np.random.choice(GENRES)
        if genre == "Comedy":
            year = int(np.clip(np.random.normal(2018, 4), 1980, 2026))
        elif genre == "Documentary":
            year = int(np.clip(np.random.normal(2005, 12), 1980, 2026))
        else:
            year = int(np.random.uniform(1980, 2026))
        rating = float(np.clip(np.random.normal(6.5, 1.2), 1.0, 10.0))
        ids.append(i)
        titles.append(f"Movie_{i}")
        genres.append(genre)
        years.append(year)
        ratings.append(round(rating, 1))
    return list(zip(ids, titles, genres, years, ratings))

# Generate synthetic actor records.
def gen_actors(n):
    ids, names, nats = [], [], []
    for i in range(n):
        nat = np.random.choice(NATIONALITIES)
        ids.append(i)
        names.append(f"Actor_{i}")
        nats.append(nat)
    return list(zip(ids, names, nats))

# Generate casting relationships.
def gen_cast(movies, actors, n):
    nat_by_id = {a[0]: a[2] for a in actors}
    genre_by_id = {m[0]: m[2] for m in movies}

    skew = {
        "Indian": {"Comedy": 3.0, "Drama": 2.5},
        "Japanese": {"Action": 2.5, "Horror": 2.0},
        "Korean": {"Action": 2.0, "Horror": 2.5},
    }

    movie_ids = [m[0] for m in movies]
    actor_ids = [a[0] for a in actors]
    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n and attempts < n * 5:
        attempts += 1
        mid = np.random.choice(movie_ids)
        genre = genre_by_id[mid]
        weights = np.ones(len(actor_ids))
        for idx, aid in enumerate(actor_ids):
            nat = nat_by_id[aid]
            mult = skew.get(nat, {}).get(genre, 1.0)
            weights[idx] = mult
        weights = weights / weights.sum()
        aid = np.random.choice(actor_ids, p=weights)
        mid, aid = int(mid), int(aid)
        key = (mid, aid)
        if key in seen:
            continue
        seen.add(key)
        rows.append((mid, aid))
    return rows

# Create and populate database.
def main():
    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)
    conn = sqlite3.connect(OUT_DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE movies (
        id INTEGER PRIMARY KEY, title TEXT, genre TEXT, year INTEGER, rating REAL)""")
    c.execute("""CREATE TABLE actors (
        id INTEGER PRIMARY KEY, name TEXT, nationality TEXT)""")
    c.execute("""CREATE TABLE casting (
        movie_id INTEGER, actor_id INTEGER)""")

    movies = gen_movies(N_MOVIES)
    actors = gen_actors(N_ACTORS)
    cast = gen_cast(movies, actors, N_CAST)

    c.executemany("INSERT INTO movies VALUES (?,?,?,?,?)", movies)
    c.executemany("INSERT INTO actors VALUES (?,?,?)", actors)
    c.executemany("INSERT INTO casting VALUES (?,?)", cast)

    c.execute("CREATE INDEX idx_cast_movie ON casting(movie_id)")
    c.execute("CREATE INDEX idx_cast_actor ON casting(actor_id)")
    conn.commit()

    print(f"movies={len(movies)} actors={len(actors)} cast={len(cast)}")
    print("DB written to", OUT_DB)
    conn.close()

# Run data generation script.
if __name__ == "__main__":
    main()