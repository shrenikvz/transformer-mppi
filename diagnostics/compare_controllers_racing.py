"""Run racing benchmark comparison (quick profile)."""

from transformer_mppi.pipelines import run_reproduction_task


if __name__ == "__main__":
    run_reproduction_task(task="racing", profile="quick", output_dir="artifacts", circuit_csv="circuit.csv")
