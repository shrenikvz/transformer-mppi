"""Run navigation2d benchmark comparison (quick profile)."""

from transformer_mppi.pipelines import run_reproduction_task


if __name__ == "__main__":
    run_reproduction_task(task="navigation2d", profile="quick", output_dir="artifacts", circuit_csv="circuit.csv")
