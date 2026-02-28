"""Legacy entrypoint for the racing transformer-MPPI pipeline."""

from transformer_mppi.pipelines import run_reproduction_task


def generate_data_and_train_transformer(
    num_episodes: int = 30,
    horizon: int = 25,
    num_samples: int = 2500,
    k: int = 5,
    t: int = 1,
    n_obstacles: int = 50,
    max_steps: int = 500,
    n_waypoints: int = 10,
    transformer_model_path: str = "model_racing.pth",
    input_scaler_path: str = "input_scaler_racing.pkl",
    output_scaler_path: str = "output_scaler_racing.pkl",
    batch_size: int = 64,
    num_epochs: int = 80,
    learning_rate: float = 5e-4,
    save_model: bool = True,
):
    # Parameters are retained for compatibility; the new package uses profile-driven config.
    del (
        num_episodes,
        horizon,
        num_samples,
        k,
        t,
        n_obstacles,
        max_steps,
        n_waypoints,
        transformer_model_path,
        input_scaler_path,
        output_scaler_path,
        batch_size,
        num_epochs,
        learning_rate,
        save_model,
    )
    run_reproduction_task(task="racing", profile="quick", output_dir="artifacts", circuit_csv="circuit.csv")


if __name__ == "__main__":
    generate_data_and_train_transformer()
