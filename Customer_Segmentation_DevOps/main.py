import os
import logging
import hydra
from omegaconf import DictConfig
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the steps of your project
_steps = [
    "loader",
    "cleaning",
    "rfm",
    "stats",
    "insights",
    "pca",
    "data_visualization"
]

@hydra.main(config_name='config')
def main(config: DictConfig):
    # Setup the wandb experiment
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    root_path = hydra.utils.get_original_cwd()

    for step in active_steps:
        try:
            # Define the directory based on the step
            if step in ["loader", "cleaning", "rfm"]:
                dir_name = "data_preparation"
            elif step in ["stats", "insights"]:
                dir_name = "descriptive_stats"
            elif step == "pca":
                dir_name = "dimensionality_reduction"
            elif step == "data_visualization":
                dir_name = "visualization"
            else:
                logger.warning(f"Unknown step: {step}")
                continue

            # Define the path to the directory containing the MLproject file
            project_path = os.path.join(root_path, "src", dir_name)

            # Run the MLflow project for the specific step
            step_params = {k: f'"{v}"' if ' ' in str(v) else v for k, v in config.get(step, {}).items()}

            _ = mlflow.run(
                project_path,
                step,  # The entry point should match the step name
                parameters=step_params
            )
        except Exception as e:
            logger.error(f"MLflow project for step '{step}' failed: {e}")
            raise

if __name__ == "__main__":
    main()
