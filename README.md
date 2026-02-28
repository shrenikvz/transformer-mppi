# Transformer-MPPI

Installable package and reproducible pipeline for **Transformer-based Model Predictive Path Integral (MPPI) control**.

This refactor provides:
- A reusable Python package API.
- End-to-end data generation, training, and benchmarking pipelines.
- Reproduction outputs (CSV metrics) aligned with the manuscript protocol.
- A `quick` profile for practical local runs and a `paper` profile for full-scale settings.

## Installation

```bash
pip install -e .
```

## Package usage

Distribution name is `transformer-mppi`; Python import path uses underscore:

```python
from transformer_mppi import TransformerMPPIController
```

### Loading a trained controller

```python
import torch
from transformer_mppi import TransformerMPPIController

controller = TransformerMPPIController.from_checkpoint(
    checkpoint_dir="artifacts/racing/checkpoints/racing_quick",
    dim_state=4,
    dim_control=2,
    dynamics=my_dynamics_fn,
    cost_func=my_cost_fn,
    u_min=torch.tensor([-2.0, -0.25]),
    u_max=torch.tensor([2.0, 0.25]),
    sigmas=torch.tensor([1.0, 1.0]),
)
```

## Reproducible pipeline

Run both tasks:

```bash
python -m transformer_mppi.cli reproduce --task both --profile quick --circuit-csv circuit.csv --output-dir artifacts
```

Run one task:

```bash
python -m transformer_mppi.cli reproduce --task navigation2d --profile quick --output-dir artifacts
python -m transformer_mppi.cli reproduce --task racing --profile quick --circuit-csv circuit.csv --output-dir artifacts
```

## Output structure

For each task (`navigation2d`, `racing`) under `output-dir`:
- `checkpoints/<task>_<profile>/`
  - `model.pt`
  - `metadata.json`
  - `input_scaler.pkl`
  - `output_scaler.pkl`
- `csv/`
  - Sample-sweep and dynamic-obstacle benchmark CSVs
  - Transformer prediction CSVs (`*_u1.csv`, `*_u2.csv`)
- `training_history.csv`

## Profiles

- `quick`: reduced training workload for fast iteration; retains paper protocol structure and trend-oriented comparisons.
- `paper`: full experiment scale from manuscript tables (long runtime).

## Legacy compatibility

Legacy scripts in `training/`, `diagnostics/`, and `src/` are retained as wrappers and now route to the package pipeline.
