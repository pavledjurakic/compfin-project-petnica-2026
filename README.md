# CompFin Project - Petnica 2026

## Project description

Project developed for the Computational Finance seminar - Petnica 2026.

## Structure

- data/ - graph data
- neural_network/ - neural network models, plots and findings
- node2vec/ - embedding model, plots and findings
- random_forest/ - model, plots and findings

## Setup

1. Clone the repository: `git clone repo_url`
2. Create a virtual environment: `python -m venv venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Data is not tracked in git. Place `dgraphfin.npz` and `subgraph_k30_N1.txt` in a `data/` folder at the repo root.

## Usage

Run any script directly from its own folder, e.g. `python lab.py` from `neural_network/`. Every script resolves paths relative to its own location, so this works regardless of where the repo is cloned.

- `neural_network/lab.py` — primary baseline (logistic regression vs MLP)
- `neural_network/experiments/` — architecture/hyperparameter comparisons
- `random_forest/random_forest.py` — Random Forest baseline
- `node2vec/embeddings.py` — graph embedding experiment

## Contributing

- Follow coding standards in the repo. Open issues and pull requests for changes.
