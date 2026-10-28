"""Optional MLflow or Weights & Biases experiment tracking adapters."""

import os

from dotenv import load_dotenv

load_dotenv()


def log_experiment(report: dict, backend: str = "mlflow") -> None:
    if backend == "mlflow":
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError("Install the MLflow extra with `pip install -e '.[ops]'`.") from error
        with mlflow.start_run():
            mlflow.log_param("algorithm", report["model"])
            mlflow.log_param("threshold", report["threshold"])
            mlflow.log_metrics(
                {
                    f"test_{key}": float(report["test_metrics"][key])
                    for key in ("precision", "recall", "f1", "average_precision", "roc_auc")
                }
            )
            mlflow.set_tag("target", report["target"])
    elif backend == "wandb":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install the W&B extra with `pip install -e '.[ops]'`.") from error
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError("Set WANDB_API_KEY before using W&B tracking.")
        run = wandb.init(project=os.getenv("WANDB_PROJECT", "finguard-ai"), config=report["split"])
        run.log({"threshold": report["threshold"], **report["test_metrics"]})
        run.finish()
    else:
        raise ValueError("Tracking backend must be 'mlflow' or 'wandb'.")
