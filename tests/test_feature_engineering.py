import pandas as pd
from api.predictor import SalesPredictor


def test_feature_engineering():
    p = SalesPredictor()
    raw = pd.DataFrame(
        [{"Store": 1, "Dept": 1, "Date": "2012-11-02", "IsHoliday": False}]
    )
    df = p.process_raw_dataframe(raw)
    assert all(col in df.columns for col in p.feature_columns)
