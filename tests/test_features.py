import numpy as np
import pandas as pd

from finguard.features import engineer_transaction_features


def test_engineer_features_adds_log_amount_and_elapsed_cycle():
    frame = pd.DataFrame({"Amount": [0.0, 9.0], "Time": [0.0, 21600.0]})

    result = engineer_transaction_features(frame)

    assert np.allclose(result["LogAmount"], [0.0, np.log(10)])
    assert result.loc[0, "ElapsedCycleSin"] == 0.0
    assert result.loc[1, "ElapsedCycleSin"] == 1.0
    assert list(frame.columns) == ["Amount", "Time"]
