from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd

from api.schemas import SingleForecastRequest, SingleForecastResponse
from api.predictor import SalesPredictor

app = FastAPI(
    title="Retail Forecasting API",
    version="1.0.0",
    description="Prediction API using the Champion machine learning model.",
)

predictor = SalesPredictor()


@app.get("/health")
def health() -> dict:
    """Performs a health check on the API and model status.

    Returns:
        dict: A dictionary containing the status and model loading state.
    """
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=SingleForecastResponse)
def predict(request: SingleForecastRequest) -> SingleForecastResponse:
    """Predicts weekly sales for a single incoming record.

    Args:
        request (SingleForecastRequest): The incoming forecast request data.

    Returns:
        SingleForecastResponse: The predicted weekly sales response.
    """
    # Use model_dump() for Pydantic v2 compatibility
    raw_df = pd.DataFrame([request.model_dump()])

    prediction = predictor.predict_single(raw_df)

    return SingleForecastResponse(
        Store=request.Store,
        Dept=request.Dept,
        Date=request.Date,
        predicted_weekly_sales=prediction,
    )


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Renders the HTML landing page for the API.

    Returns:
        str: An HTML formatted string containing welcome information and endpoint links.
    """
    return """
    <html>
        <head>
            <title>Retail Forecasting API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f7f7f7;
                    padding: 40px;
                    color: #333;
                }
                .container {
                    max-width: 600px;
                    margin: auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                }
                a {
                    color: #2980b9;
                    text-decoration: none;
                    font-weight: bold;
                }
                a:hover {
                    text-decoration: underline;
                }
                .endpoint {
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Retail Forecasting API</h1>
                <p>Your API is running successfully.</p>

                <div class="endpoint">
                    <p>📘 Interactive documentation:</p>
                    <a href="/docs">/docs</a>
                </div>

                <div class="endpoint">
                    <p>📗 Alternative documentation:</p>
                    <a href="/redoc">/redoc</a>
                </div>

                <div class="endpoint">
                    <p>💓 Healthcheck:</p>
                    <a href="/health">/health</a>
                </div>

                <p style="margin-top:30px; font-size:12px; color:#777;">
                    Retail Forecasting API · Powered by FastAPI
                </p>
            </div>
        </body>
    </html>
    """
