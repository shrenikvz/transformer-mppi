"""Legacy entrypoint for the navigation2d transformer-MPPI pipeline."""

from transformer_mppi.pipelines import run_reproduction_task


def run_experiments(
    num_epochs: int = 80,
    batch_size: int = 128,
    fixed_maps: int = 40,
    error_thresholds: list[float] | None = None,
    k: int = 5,
    output_dir: str = "artifacts",
):
    # Parameters are retained for compatibility; the new package uses profile-driven config.
    del num_epochs, batch_size, fixed_maps, error_thresholds, k
    run_reproduction_task(task="navigation2d", profile="quick", output_dir=output_dir)


if __name__ == "__main__":
    run_experiments()
