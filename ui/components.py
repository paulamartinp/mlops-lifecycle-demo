import gradio as gr


def build_header():
    return gr.Markdown("""
        <div id="title">
        # 🛒 Walmart Sales Forecasting
        </div>

        <div id="subtitle">
        Predict weekly sales using the current **Champion Model**
        registered in the **MLflow Model Registry**.
        </div>
    """)


def build_model_info():
    with gr.Accordion("📊 Model Information", open=False) as acc:
        gr.Markdown("""
            **Architecture**

            - Training Pipeline ✅
            - MLflow Experiment Tracking ✅
            - MLflow Model Registry ✅
            - FastAPI Serving ✅
            - Gradio UI ✅

            Predictions are generated using the current
            **Champion Model** from the MLflow Registry.
        """)

    return acc


def build_inputs():
    gr.Markdown("## Forecast Inputs")

    with gr.Row():
        store = gr.Number(label="🏪 Store", value=1, precision=0)
        dept = gr.Number(label="📦 Department", value=1, precision=0)

    with gr.Row():
        date = gr.Textbox(label="📅 Date (YYYY-MM-DD)", value="2012-11-23")
        is_holiday = gr.Checkbox(label="🎄 Holiday Week", value=False)

    return store, dept, date, is_holiday


def build_predict_button():
    return gr.Button("🚀 Predict Sales", variant="primary", size="lg")


def build_output():
    gr.Markdown("## 💰 Forecast Result")
    return gr.Textbox(label="Predicted Weekly Sales", interactive=False)
