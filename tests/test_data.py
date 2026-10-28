import pandas as pd
import pytest

from finguard.data import REQUIRED_COLUMNS, load_transactions, transaction_summary


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {**{column: 0.0 for column in REQUIRED_COLUMNS}, "Class": 0},
            {**{column: 1.0 for column in REQUIRED_COLUMNS}, "Class": 1},
        ]
    )


def test_load_transactions_validates_and_preserves_binary_target(tmp_path):
    path = tmp_path / "transactions.csv"
    valid_frame().to_csv(path, index=False)

    frame = load_transactions(path)

    assert len(frame) == 2
    assert frame["Class"].tolist() == [0, 1]
    assert transaction_summary(frame)["fraud_transactions"] == 1


def test_load_transactions_rejects_missing_columns(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame({"Amount": [1.0], "Class": [0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_transactions(path)


def test_load_transactions_rejects_missing_values(tmp_path):
    frame = valid_frame()
    frame.loc[0, "Amount"] = float("nan")
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_transactions(path)


def test_load_transactions_reports_but_preserves_duplicate_rows(tmp_path):
    frame = valid_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)

    loaded = load_transactions(path)

    assert len(loaded) == 3
    assert transaction_summary(loaded)["duplicate_rows"] == 1
