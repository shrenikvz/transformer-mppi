"""Legacy entrypoint retained for compatibility.

Runs both quick-profile pipelines.
"""

from transformer_mppi.pipelines import run_reproduction


if __name__ == "__main__":
    run_reproduction(task="both", profile="quick", output_dir="artifacts", circuit_csv="circuit.csv")
