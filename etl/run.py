import os
import pandas as pd
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "./data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
CSV_PATH = os.path.join(DATA_DIR, "sales_data.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "sales_data.parquet")
DUCKDB_PATH = os.path.join(DATA_DIR, "sales.duckdb")

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    return df

def save_parquet(df, path):
    try:
        df.to_parquet(path, index=False)
        print("Parquet written:", path)
    except Exception as e:
        print("Parquet failed:", e)

def save_duckdb(df, path):
    try:
        import duckdb
        con = duckdb.connect(path)
        con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM df")
        con.close()
        print("DuckDB written:", path)
    except Exception as e:
        print("DuckDB failed:", e)

def main():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found")
    df = load_csv(CSV_PATH)
    save_parquet(df, PARQUET_PATH)
    save_duckdb(df, DUCKDB_PATH)
    print("ETL complete.")

if __name__ == "__main__":
    main()
