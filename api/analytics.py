# !-- analytics.py - Retired --!
# Consolidated into the main app for simplicity.

# import os
# import pandas as pd

# DATA_DIR = os.environ.get("DATA_DIR", "./data")

# def load_df():
#     duckdb_file = os.path.join(DATA_DIR, "sales.duckdb")
#     parquet = os.path.join(DATA_DIR, "sales_data.parquet")
#     csv = os.path.join(DATA_DIR, "sales_data.csv")

#     df = None
#     try:
#         import duckdb
#         con = duckdb.connect(duckdb_file)
#         df = con.execute("SELECT * FROM sales").df()
#         con.close()
#     except Exception:
#         if os.path.exists(parquet):
#             try:
#                 df = pd.read_parquet(parquet)
#             except Exception:
#                 pass
#         if df is None and os.path.exists(csv):
#             df = pd.read_csv(csv)

#     if df is None:
#         raise FileNotFoundError("No dataset found in data/ (duckdb/parquet/csv)")

#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     date_col = next((c for c in df.columns if "date" in c.lower()), None)
#     if date_col:
#         df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
#     return df

# def kpis():
#     df = load_df()
#     metric_col = next((c for c in df.columns if c.lower() in ("sales","revenue","amount","total","qty","quantity")), None)
#     region_col = next((c for c in df.columns if "region" in c.lower()), None)
#     product_col = next((c for c in df.columns if "product" in c.lower()), None)
#     sat_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)

#     total_sales = float(df[metric_col].sum()) if metric_col else None
#     avg_satisfaction = float(df[sat_col].mean()) if sat_col else None

#     top_region = None
#     if region_col and metric_col:
#         grp = df.groupby(region_col)[metric_col].sum().sort_values(ascending=False)
#         if not grp.empty:
#             top_region = grp.index[0]

#     top_product = None
#     if product_col and metric_col:
#         grp = df.groupby(product_col)[metric_col].sum().sort_values(ascending=False)
#         if not grp.empty:
#             top_product = grp.index[0]

#     return {
#         "total_sales": total_sales,
#         "avg_satisfaction": avg_satisfaction,
#         "top_region": top_region,
#         "top_product": top_product,
#     }

# def forecast_baseline(horizon=14):
#     df = load_df()
#     metric_col = next((c for c in df.columns if c.lower() in ("sales","revenue","amount","total","qty","quantity")), None)
#     date_col = next((c for c in df.columns if "date" in c.lower()), None)
#     if not (metric_col and date_col):
#         return {"message": "No date/metric present", "series": []}

#     ts = df[[date_col, metric_col]].dropna().copy().sort_values(date_col)
#     daily = ts.groupby(ts[date_col].dt.date)[metric_col].sum().reset_index()
#     daily.columns = ["date","y"]
#     daily["date"] = pd.to_datetime(daily["date"])

#     # Simple MA7 baseline
#     daily["ma7"] = daily["y"].rolling(7, min_periods=1).mean()
#     last_date = daily["date"].max()
#     last_ma = float(daily["ma7"].iloc[-1])

#     future = []
#     for i in range(1, horizon+1):
#         future.append({"date": (last_date + pd.Timedelta(days=i)).date().isoformat(), "yhat": last_ma})

#     return {
#         "model": "ma7_baseline",
#         "history": daily.tail(60).to_dict(orient="records"),
#         "forecast": future
#     }

# from typing import List, Dict, Any

# def region_divergence(con) -> List[Dict[str, Any]]:
#     # Use epoch(time) as x for regression; DuckDB: epoch(TIMESTAMP) → seconds
#     sql = """
#     WITH s AS (
#       SELECT
#         region,
#         epoch(CAST(date AS TIMESTAMP)) AS x,
#         sales::DOUBLE                 AS y_sales,
#         satisfaction::DOUBLE          AS y_sat
#       FROM sales
#     )
#     SELECT
#       region,
#       regr_slope(y_sales, x)       AS slope_sales,
#       regr_slope(y_sat,   x)       AS slope_sat,
#       COUNT(*)                      AS n
#     FROM s
#     GROUP BY region
#     HAVING slope_sales > 0 AND slope_sat < 0
#     ORDER BY slope_sales DESC;
#     """
#     rows = con.execute(sql).fetchall()
#     cols = [d[0] for d in con.description]
#     return [dict(zip(cols, r)) for r in rows]

# def top_products_under_30(con, limit:int=2) -> List[Dict[str, Any]]:
#     sql = f"""
#     SELECT
#       product,
#       SUM(sales)::DOUBLE AS total_sales,
#       COUNT(*)           AS n
#     FROM sales
#     WHERE age < 30
#     GROUP BY product
#     ORDER BY total_sales DESC
#     LIMIT {int(limit)};
#     """
#     rows = con.execute(sql).fetchall()
#     cols = [d[0] for d in con.description]
#     return [dict(zip(cols, r)) for r in rows]

# def region_monthly_trends(con, regions: List[str]) -> List[Dict[str, Any]]:
#     # Monthly aggregate for charts
#     if not regions:
#         return []
#     # build safe IN list
#     safe = ",".join("'" + r.replace("'", "''") + "'" for r in regions)
#     sql = f"""
#     SELECT
#       date_trunc('month', date)::DATE AS month,
#       region,
#       SUM(sales)::DOUBLE              AS sales,
#       AVG(satisfaction)::DOUBLE       AS satisfaction
#     FROM sales
#     WHERE region IN ({safe})
#     GROUP BY 1,2
#     ORDER BY 1,2;
#     """
#     rows = con.execute(sql).fetchall()
#     cols = [d[0] for d in con.description]
#     return [dict(zip(cols, r)) for r in rows]