import pandas as pd
from api.predictor import SalesPredictor


def test_predict_single():
    p = SalesPredictor()
    raw = pd.DataFrame(
        [{"Store": 1, "Dept": 1, "Date": "2012-11-02", "IsHoliday": False}]
    )
    pred = p.predict_single(raw)
    assert isinstance(pred, float)
