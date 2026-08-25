from pydantic import BaseModel, Field


class SingleForecastRequest(BaseModel):
    """Schema representing an incoming prediction request for a single record.

    Attributes:
        Store (int): Store identification number.
        Dept (int): Department identification number.
        Date (str): Target date in YYYY-MM-DD format.
        IsHoliday (bool): Flag indicating whether the week includes a holiday.
    """

    Store: int = Field(..., example=1, description="Store identification number")
    Dept: int = Field(..., example=1, description="Department identification number")
    Date: str = Field(
        ..., example="2012-11-02", description="Target date in YYYY-MM-DD format"
    )
    IsHoliday: bool = Field(
        ...,
        example=False,
        description="Flag indicating whether the week includes a holiday",
    )


class SingleForecastResponse(BaseModel):
    """Schema representing the prediction response for a single record.

    Attributes:
        Store (int): Store identification number.
        Dept (int): Department identification number.
        Date (str): Target date in YYYY-MM-DD format.
        predicted_weekly_sales (float): Predicted weekly sales amount.
    """

    Store: int = Field(..., example=1, description="Store identification number")
    Dept: int = Field(..., example=1, description="Department identification number")
    Date: str = Field(
        ..., example="2012-11-02", description="Target date in YYYY-MM-DD format"
    )
    predicted_weekly_sales: float = Field(
        ..., example=15430.50, description="Predicted weekly sales amount"
    )
