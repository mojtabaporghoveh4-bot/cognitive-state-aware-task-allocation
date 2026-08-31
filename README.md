# Cognitive-State-Aware Rolling-Horizon Task Allocation

This repository contains the simulation and reporting code for cognitive-state-aware task allocation in a multirobot supervision setting. The proposed CSA-RH method assigns tasks to robots and operators while accounting for task priorities, deadlines, robot eligibility, robot availability, operator stress, fatigue, impairment, and state-dependent supervision capacity.

The repository also includes the saved results of the reference experiment. This lets readers regenerate all report figures without repeating the longer hyperparameter-tuning run.

## What is included

| Path | Purpose |
| --- | --- |
| `src/msa_ta_paper_model.py` | Defines tasks, robots, operators, scenarios, state dynamics, feasibility checks, cost functions, allocation methods, hyperparameter tuning, simulation, and evaluation metrics. |
| `src/msa_ta_clean_report.py` | Reads experiment outputs, creates summary tables, and generates the complete set of figures as PDF files. |
| `reference_results/` | Saved CSV, JSON, and text outputs from the reference experiment: 10 final runs per scenario, 20 Optuna trials per scenario, and 3 tuning seeds. |
| `requirements.txt` | Python dependencies. |
| `CITATION.cff` | Citation metadata for this research code. |

## Allocation methods

The experiment compares five methods:

- **Random:** randomly orders available tasks and constructs a feasible assignment.
- **Greedy:** prioritizes tasks by decreasing priority, then by deadline and processing time.
- **Myopic exact-step:** solves the current one-step assignment using OR-Tools CP-SAT when available, with a one-step beam-search fallback.
- **Cognitive-unaware RH:** uses rolling-horizon beam search without the cognitive objective or cognitive-state-dependent capacity constraint.
- **Proposed CSA-RH:** uses rolling-horizon beam search with stress, fatigue, impairment, mental-state belief, and belief-dependent operator capacity.

For compatibility with the saved data, the last two methods retain the internal identifiers `human_unaware_rh` and `proposed_msa_rh`.

## Scenarios

- **Feasible load:** the available team capacity is sufficient for most work.
- **Overload finite shift:** the finite shift cannot accommodate every task.
- **Dynamic disturbance:** tasks arrive during execution, one robot becomes temporarily unavailable, and one operator's capacity is reduced.

## Installation

Python 3.12 was used for the reference experiment.

```powershell
python -m pip install -r requirements.txt
```

## Generate all figures from the saved results

Run this command from the repository root:

```powershell
python src/msa_ta_clean_report.py
```

The script reads `reference_results/` by default and writes:

- PDF figures to `generated_report/figures/`
- summary CSV files to `generated_report/`
- a short index to `generated_report/results_report.md`

Only PDF figures are generated. The reporting script covers all available figure groups, including performance metrics, evaluation and instantaneous cost decompositions, cost trajectories, mental-state distributions, operator impairment trajectories, cumulative impairment, and unfinished-task diagnostics.

Custom input and output folders can be supplied explicitly:

```powershell
python src/msa_ta_clean_report.py --input-dir my_results --output-dir my_report
```

## Run a quick model check

```powershell
python src/msa_ta_paper_model.py --toy
```

This runs a small built-in problem and prints the selected actions and evaluation metrics.

## Reproduce the full reference experiment

The following command tunes the model separately for each scenario and then performs 10 final runs:

```powershell
python src/msa_ta_paper_model.py `
  --output-dir reproduced_results `
  --runs 10 `
  --seed 123 `
  --tune `
  --tune-trials 20 `
  --tune-seed-count 3
```

Then generate the figures:

```powershell
python src/msa_ta_clean_report.py `
  --input-dir reproduced_results `
  --output-dir reproduced_report
```

The full run is computationally heavier than generating figures from `reference_results/`. Runtime depends on the machine and on the scenario-specific beam width and candidate-action limit selected during tuning.

## Optimization notes

The full rolling-horizon model is nonlinear because stress, fatigue, impairment, mental-state beliefs, and belief-weighted capacity evolve through future assignment decisions. The two rolling-horizon methods therefore use beam search over feasible candidate actions; their solutions are approximate and are not claimed to be globally optimal.

The myopic method solves only the current one-step binary assignment exactly with CP-SAT when OR-Tools is available. Optuna uses a Tree-structured Parzen Estimator sampler to tune model and search-control hyperparameters separately for each scenario.

## Main reported metrics

- `TWT`: total weighted tardiness, lower is better.
- `DMR`: deadline miss rate, lower is better.
- `WCR`: weighted completion ratio, higher is better.
- `CI`: cumulative impairment, lower is better.
- `PI`: peak impairment, lower is better.

## Reproducibility

Random seeds are saved in `detailed_run_metrics.csv`. Selected scenario-specific hyperparameters are stored in `best_hyperparameters_by_scenario.json`, while the complete Optuna histories are stored in the three `tuning_trials_*.csv` files.

## License

This project is released under the MIT License. See `LICENSE`.
