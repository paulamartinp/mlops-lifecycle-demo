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



DROP VIEW IF EXISTS train_dataset;
DROP VIEW IF EXISTS test_dataset;

-- Train
CREATE VIEW train_dataset AS
SELECT
    s.Store,
    s.Dept,
    s.Date,
    s.Weekly_Sales,
    s.IsHoliday,
    f.Temperature,
    f.Fuel_Price,
    f.MarkDown1,
    f.MarkDown2,
    f.MarkDown3,
    f.MarkDown4,
    f.MarkDown5,
    f.CPI,
    f.Unemployment,
    st.Type AS Store_Type,
    st.Size AS Store_Size
FROM sales s
LEFT JOIN features f
    ON s.Store = f.Store
   AND s.Date = f.Date
LEFT JOIN stores st
    ON s.Store = st.Store;

-- Test
CREATE VIEW test_dataset AS
SELECT
    t.Store,
    t.Dept,
    t.Date,
    t.IsHoliday,
    f.Temperature,
    f.Fuel_Price,
    f.MarkDown1,
    f.MarkDown2,
    f.MarkDown3,
    f.MarkDown4,
    f.MarkDown5,
    f.CPI,
    f.Unemployment,
    st.Type AS Store_Type,
    st.Size AS Store_Size
FROM test t
LEFT JOIN features f
    ON t.Store = f.Store
   AND t.Date = f.Date
LEFT JOIN stores st
    ON t.Store = st.Store;