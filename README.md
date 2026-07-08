# Cardinality Estimator

A lightweight project that explores how machine learning can improve SQL query optimization by producing better cardinality estimates than traditional histogram-based methods.

The project implements a small in-memory query engine, a cost-based optimizer, and two different cardinality estimators. It then compares how the optimizer behaves when driven by each estimator using a synthetic movie database.

---

## Features

- In-memory SQL execution engine
- Histogram-based cardinality estimator
- Learned cardinality estimator using XGBoost
- Cost-based join order optimization
- Hash Join and Nested Loop Join support
- Synthetic benchmark dataset with correlated attributes
- Automatic workload generation
- End-to-end benchmarking and visualization

---

## Project Structure

```
Cardinality-Estimator/
│
├── engine/
│   ├── catalog.py
│   ├── executor.py
│   └── ir.py
│
├── estimators/
│   ├── histogram.py
│   ├── learned.py
│   └── features.py
│
├── planner/
│   ├── cost_model.py
│   └── join_optimizer.py
│
├── data/
│   └── generate_data.py
│
├── benchmark/
│   ├── generate_queries.py
│   ├── run_benchmark.py
│   └── report.py
│
├── results/
├── requirements.txt
└── README.md
```

---

## How it Works

The overall pipeline is

```
Synthetic Database
        │
        ▼
Random Query Generator
        │
        ▼
Histogram Estimator
        │
        ├──────────────┐
        ▼              │
 Learned Estimator     │
        │              │
        └──────┬───────┘
               ▼
      Query Optimizer
               ▼
        Execution Engine
               ▼
 Performance Comparison
```

---

## Components

### Query Engine

The engine loads database tables into memory using pandas DataFrames.

It supports

- Selection predicates
- Multi-table joins
- Hash Join
- Nested Loop Join

---

### Histogram Estimator

Uses per-column histograms to estimate predicate selectivity.

Join cardinalities are estimated using the classical

```
|R ⋈ S| ≈ |R| × |S| / max(V(R), V(S))
```

formula under the assumption that predicates are independent.

---

### Learned Estimator

A gradient boosted regression model (XGBoost) is trained to predict query cardinalities.

Input features include

- Predicate types
- Predicate values
- Join structure
- Query shape
- Histogram estimate

The histogram estimate is also used as an input feature, allowing the model to learn corrections instead of predicting from scratch.

---

### Query Planner

The planner enumerates valid join orders and estimates the execution cost for each one.

For every join step it chooses between

- Nested Loop Join
- Hash Join

using a simple cost model.

The cheapest plan is selected for execution.

---

### Benchmark

The benchmark evaluates both estimators by comparing

- Cardinality estimation accuracy
- Query q-error
- Planner decisions
- End-to-end execution time

The benchmark also generates plots to visualize the results.

---

## Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Generate the synthetic database

```bash
python data/generate_data.py
```

Run the benchmark

```bash
python benchmark/run_benchmark.py
```

Generate result plots

```bash
python benchmark/report.py
```

---

## Technologies

- Python
- SQLite
- Pandas
- NumPy
- XGBoost
- Matplotlib

---
