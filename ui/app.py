"""Gradio UI for Walmart Sales Forecasting."""

import gradio as gr
import pandas as pd

from api.predictor import SalesPredictor
from ui.styles import CUSTOM_CSS
from ui.components import (
    build_header,
    build_model_info,
    build_inputs,
    build_output,
    build_predict_button,
)

predictor = SalesPredictor()


def predict_sales(
    store: int,
    dept: int,
    date: str,
    is_holiday: bool,
) -> str:
    """Generate a sales forecast."""

    raw_df = pd.DataFrame(
        [
            {
                "Store": store,
                "Dept": dept,
                "Date": date,
                "IsHoliday": is_holiday,
            }
        ]
    )

    prediction = predictor.predict_single(raw_df)

    return f"${prediction:,.2f}"


with gr.Blocks(title="Walmart Sales Forecasting") as demo:

    build_header()
    build_model_info()

    store, dept, date, is_holiday = build_inputs()
    predict_button = build_predict_button()
    prediction_output = build_output()

    predict_button.click(
        fn=predict_sales,
        inputs=[store, dept, date, is_holiday],
        outputs=prediction_output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
