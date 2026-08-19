DROP VIEW IF EXISTS forecasting_dataset;

CREATE VIEW forecasting_dataset AS

SELECT
    s.Store,
    s.Date,
    s.Weekly_Sales,
    s.IsHoliday,
    f.Temperature,
    f.Fuel_Price,
    f.CPI,
    f.Unemployment,
    st.Type,
    st.Size

FROM sales s

LEFT JOIN features f
    ON s.Store = f.Store
    AND s.Date = f.Date

LEFT JOIN stores st
    ON s.Store = st.Store;