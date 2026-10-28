import pandas as pd
import pytest

from finguard.credit import train_credit_model


def test_credit_model_requires_external_binary_target(tmp_path):
    source = tmp_path / "credit.csv"
    pd.DataFrame(
        {
            "income": range(30),
            "segment": ["a", "b"] * 15,
            "defaulted": [0, 1] * 15,
        }
    ).to_csv(source, index=False)

    result = train_credit_model(source, "defaulted", tmp_path / "model")

    assert result["target"] == "defaulted"
    assert (tmp_path / "model" / "credit_model.joblib").is_file()
    assert "fairness" in result["warning"]


def test_credit_model_rejects_missing_target(tmp_path):
    source = tmp_path / "credit.csv"
    pd.DataFrame({"income": [1, 2], "Class": [0, 1]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="not present"):
        train_credit_model(source, "defaulted", tmp_path / "model")
