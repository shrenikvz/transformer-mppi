"""Legacy entrypoint retained for compatibility.

Use `python -m transformer_mppi.cli reproduce --task racing --profile quick` for the new pipeline.
"""

from training.main_transformer_autonomous_racing_new_mppi import generate_data_and_train_transformer


if __name__ == "__main__":
    generate_data_and_train_transformer()
