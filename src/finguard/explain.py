"""Optional SHAP explanation helpers for tabular fraud models."""

import importlib.util

import pandas as pd


def explain_instance(model, row: pd.DataFrame, background: pd.DataFrame) -> list[dict]:
    if importlib.util.find_spec("shap") is None:
        raise RuntimeError("SHAP is optional; install with `pip install -e '.[explain]'`.")
    import shap

    classifier = model
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        scaler = model.named_steps.get("scale")
        if scaler is not None:
            row = pd.DataFrame(scaler.transform(row), columns=row.columns, index=row.index)
            background = pd.DataFrame(
                scaler.transform(background), columns=background.columns, index=background.index
            )
    explainer = shap.Explainer(classifier, background)
    values = explainer(row).values
    contributions = values[0]
    if contributions.ndim > 1:
        contributions = contributions[:, -1]
    ranked = sorted(zip(row.columns, contributions), key=lambda item: abs(float(item[1])), reverse=True)
    return [{"feature": name, "contribution": float(value)} for name, value in ranked[:10]]
