# 🧪 Classical ML Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-261230?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Task](https://img.shields.io/badge/Task-29BEB0?logo=task&logoColor=white)](https://taskfile.dev/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=black)](https://pre-commit.com/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-FE5196?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)

**Classical ML Lab** is a practice repository for **classical machine learning**. It pairs hands-on **Kaggle competition** notebooks with study material on classical algorithms — from KNN, decision trees, and linear models to ensemble methods — built on **scikit-learn**, the **gradient-boosting** trio (**XGBoost / CatBoost / LightGBM**), and **Optuna** for tuning.

> 💡 Deep learning lives in a separate repository — this lab is intentionally scoped to classical ML.

## ✨ What's Inside

* **📚 Learning notebooks** — algorithms explained and implemented from the ground up, plus metrics and core libraries.
* **🏆 Competition notebooks** — end-to-end solutions for Kaggle competitions (Titanic, Spaceship Titanic).
* **🧰 Reusable helpers** — a small `src/` package with evaluation and plotting utilities shared across notebooks.
* **📝 Templates** — starter notebooks so every new algorithm or competition follows the same layout.

## 📂 Project Structure

```text
.
├── learning/                   # Study material
│   ├── algorithms/             # Algorithm walkthroughs (classification + regression)
│   │   ├── decision_trees/
│   │   ├── knn/                # Brute force + Annoy / HNSW / k-d trees / LSH
│   │   ├── regression_algorithms/   # Linear & logistic regression
│   │   ├── ensemble_methods/   # Bagging, random forest, gradient boosting, stacking
│   │   └── hyperparameter_tuning/   # Grid search, random search
│   ├── metrics/                # Classification & regression metrics
│   └── libraries/              # Tooling notebooks (matplotlib, optuna)
├── competitions/               # Kaggle solutions (each with its own data/ folder)
│   ├── titanic/
│   └── spaceship_titanic/
├── src/                        # Reusable helpers imported by the notebooks
│   ├── evaluations/            # Metric tables for (multi)classification & regression
│   └── plots/                  # Confusion matrices, ROC/PR curves, boundaries, etc.
├── templates/                  # Starter notebooks (algorithm.ipynb, competition.ipynb)
├── pyproject.toml              # Dependencies, ruff, mypy, commitizen config
└── Taskfile.yml                # Common developer commands
```

## 📦 Dependencies

* [Python 3.13+](https://www.python.org/downloads/)
* [uv](https://docs.astral.sh/uv/) — environment & dependency management
* [Task](https://taskfile.dev/) — task runner
* [Jupyter](https://jupyter.org/) — running the notebooks

Python packages are split into [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) in `pyproject.toml`:

| Group      | Purpose                                                        |
| ---------- | ------------------------------------------------------------- |
| `data`     | numpy, pandas                                                 |
| `ml`       | scikit-learn, XGBoost, CatBoost, LightGBM                     |
| `viz`      | matplotlib, seaborn, plotly                                   |
| `tuning`   | optuna, hyperopt, scikit-optimize                             |
| `notebook` | ipywidgets, tqdm, and other notebook UX helpers              |
| `dev`      | ruff, mypy, pytest, pre-commit, commitizen, audit tooling     |

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/NKTKLN/classical-ml-lab.git
cd classical-ml-lab

# 2. Install dependencies and git hooks
task init

# 3. Launch Jupyter and open any notebook
uv run jupyter lab
```

> `task init` runs `uv sync --all-groups` and installs the pre-commit hooks.
> Don't have [Task](https://taskfile.dev/)? Run `uv sync --all-groups` and `uv run pre-commit install --install-hooks` directly.

## 🛠️ Common Tasks

Run `task --list` to see everything. The most useful commands:

| Command          | Description                                       |
| ---------------- | ------------------------------------------------- |
| `task init`      | Full setup: sync dependencies + install hooks     |
| `task sync`      | Sync dependencies with uv                         |
| `task fmt`       | Auto-fix lint issues and format code              |
| `task lint`      | Run ruff + format check + mypy                    |
| `task audit`     | Security audit of dependencies (`pip-audit`)      |
| `task check`     | Full quality gate (lint, tests, audit, deps)      |
| `task cz-commit` | Commit using Conventional Commits                 |

## 📜 License

This project is licensed under the MIT License. See the [LICENSE.md](./LICENSE.md) file for details.
