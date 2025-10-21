import os, asyncio, textwrap, json, requests, logging, re
from typing import List, Dict, Any, Tuple, Literal, Optional
from datetime import date as _date, timedelta, datetime
from pydantic import BaseModel
from fastapi import APIRouter, FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from functools import lru_cache
import itertools

log = logging.getLogger("uvicorn.error")

try:
    import duckdb
except Exception:
    duckdb = None  # make a nice error if missing

# ---- App/version
APP_VERSION = "0.1.0"
app = FastAPI(title="Intellect BI API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Include a router decision endpoint
router = APIRouter(prefix="/router", tags=["router"])

# ---- Paths / config
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_DIR      = os.environ.get("CHROMA_PERSIST_DIR", "./vector")
DATA_DIR        = os.environ.get("DATA_DIR", "./data")
ARTIFACTS_DIR   = os.environ.get("ARTIFACTS_DIR", "./artifacts")
SALES_CSV       = os.environ.get("SALES_CSV", "./data/sales_data.csv")
DATA_TABLE      = os.environ.get("DATA_TABLE", "sales")

# ----- Router payload models -----
class RouteRequest(BaseModel):
    query: str

class RouteResponse(BaseModel):
    route: Literal["docs", "duckdb"]
    route_reason: str
    source_used: Literal["docs", "duckdb", "llm-sql", "fallback"]
    sql: Optional[str] = None

# ---- datetime corrections for DuckDB
def _to_datestr(d) -> str:
    # Accept datetime.date or datetime.datetime and return YYYY-MM-DD
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, _date):
        return d.isoformat()
    return str(d)

# ---- Deterministic router (single source of truth) ----

def _decide_route(q: str) -> Dict[str, str]:
    ql = q.lower().strip()
    doc_hit = any(w in ql for w in _DOC_WORDS)
    tab_hit = any(w in ql for w in _TABULAR_WORDS)
    if tab_hit and not doc_hit:
        return {"route": "duckdb", "route_reason": "tabular/metrics keywords detected"}
    if doc_hit and not tab_hit:
        return {"route": "docs", "route_reason": "document/summary keywords detected"}
    # tie-break: prefer duckdb for analytics phrasing
    return {"route": "duckdb", "route_reason": "tie-break → analytics default"}

@router.post("/route", response_model=RouteResponse)
def post_route(body: RouteRequest) -> RouteResponse:
    r = _decide_route(body.query)
    return RouteResponse(route=r["route"], route_reason=r["route_reason"], source_used=r["route"])

@router.get("/route", response_model=RouteResponse)
def get_route(query: str = Query(..., alias="query")) -> RouteResponse:
    r = _decide_route(query)
    return RouteResponse(route=r["route"], route_reason=r["route_reason"], source_used=r["route"])

# Register router jjust once
app.include_router(router)

@app.get("/debug/routes")
def debug_routes():
    return [
        {
            "path": r.path,
            "name": getattr(r, "name", None),
            "methods": sorted(list(getattr(r, "methods", set())))
        }
        for r in app.routes
    ]

# ---- SOme Route helper constants

SALES_SCHEMA_COLUMNS = {"date","region","product","age","gender","sales","satisfaction"}
_NUMERIC_CUES = (
    "sum","avg","average","median","min","max","total","top","rank","trend",
    "increase","decrease","growth","decline","yoy","y/y","mom","m/m","qoq","q/q",
    "quarter","month","weekly","highest","lowest","compare","correlation","corr",
    "distribution","bucket","percentile","quartile","std","variance",
    "by region","by product","by age","by gender","segment","breakdown"
)
_SALES_TERMS = (
    "sales","revenue","txn","transaction","transaction value",
    "customers","customer","satisfaction","nps","age","gender","region","product"
)
_BI_PATTERNS = (
    r"\b(top|best|worst)\b",
    r"\b(change|delta|difference|improvement|decline)\b",
    r"\b(project|forecast|estimate|predict)\b",
)
_DOC_HINTS = (
    "pdf", "document", "doc", "paper", "report", "whitepaper",
    "page ", "section ", "figure ", "table ",
    "summarize", "summary", "key ideas", "key takeaways",
    "according to", "from the pdf", "cite", "citation"
)
SAFE_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
FORBIDDEN_SQL_PATTERNS = (
    r"\b(insert|update|delete|drop|alter|truncate|create|attach|detach|copy|load)\b",
    r";\s*--",  # attempt to chain
)
_DOC_WORDS = {"pdf", "document", "doc", "summarize", "explain", "from the walmart pdf", "whitepaper", "paper"}
_TABULAR_WORDS = {"csv", "table", "quarter", "region", "avg", "average", "sum", "trend", "growth", "satisfaction", "sales"}
_METRIC_SAT = {"satisfaction", "csat"}
_METRIC_SALES = {"sales", "revenue", "transaction value", "transaction_value", "amount"}
_DIM_CANDIDATES = ("region", "product", "gender", "age")  # update if you add more dims

_TIME_GRAINS = {
    "monthly": "month",
    "per month": "month",
    "by month": "month",
    "quarterly": "quarter",
    "per quarter": "quarter",
    "by quarter": "quarter",
    "yearly": "year",
    "annual": "year",
}

_COMPARE_TOKENS = {
    "last quarter": ("quarter", "last"),
    "previous quarter": ("quarter", "previous"),
    "two most recent quarters": ("quarter", "last2"),
    "two latest quarters": ("quarter", "last2"),
    "yoy": ("year", "yoy"),
    "year-over-year": ("year", "yoy"),
}
# ---- Chroma
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
def get_docs_collection():
    try:
        return client.get_collection("docs")
    except Exception:
        return client.create_collection(name="docs", metadata={"hnsw:space":"cosine"})

# ===================== DuckDB: data-aware helpers =====================
_duck = None
_schema_cols_lower: set = set()

def _resolve_sales_csv() -> str:
    env_path = os.environ.get("SALES_CSV", "").strip()
    candidates = [p for p in [env_path, "/app/data/sales_data.csv", "./data/sales_data.csv"] if p]
    for p in candidates:
        if os.path.exists(p):
            return p
    return env_path or "./data/sales_data.csv"

# --- YoY-by-quarter Fix (force DuckDB)
def _has_yoy_quarter(q: str) -> bool:
    ql = q.lower()
    has_yoy = ("year over year" in ql) or ("yoy" in ql) or ("y-o-y" in ql) or ("y/y" in ql)
    has_qtr = ("quarter" in ql) or any(t in ql for t in ("q1", "q2", "q3", "q4", "quarterly"))
    has_sales = any(t in ql for t in ("sales", "revenue", "amount", "transaction value"))
    return has_yoy and has_qtr and has_sales

def _ensure_duckdb():
    """
    Initialize DuckDB and expose a stable, normalized view named in DATA_TABLE.
    Final schema (all lower snake-case):
      date (DATE), product (TEXT), region (TEXT),
      sales (DOUBLE), age (INT), gender (TEXT), satisfaction (DOUBLE)
    """
    global _duck, _schema_cols_lower
    if duckdb is None:
        raise HTTPException(status_code=500, detail="duckdb not installed in API image")
    if _duck is None:
        _duck = duckdb.connect()

        resolved_csv = _resolve_sales_csv()
        path_sql = resolved_csv.replace("'", "''")

        _duck.execute(f"""
        CREATE OR REPLACE VIEW __raw_sales AS
        SELECT * FROM read_csv_auto('{path_sql}', header=TRUE);
        """)

        cols = [r[1] for r in _duck.execute("PRAGMA table_info('__raw_sales')").fetchall()]
        lower = {c.lower(): c for c in cols}
        def has(name): return name.lower() in lower

        if all(has(c) for c in [
            "Date","Product","Region","Sales","Customer_Age","Customer_Gender","Customer_Satisfaction"
        ]):
            _duck.execute(f"""
            CREATE OR REPLACE VIEW {DATA_TABLE} AS
            SELECT
              CAST({lower['date']} AS DATE)                       AS date,
              {lower['product']}                                  AS product,
              {lower['region']}                                   AS region,
              CAST({lower['sales']} AS DOUBLE)                    AS sales,
              CAST({lower['customer_age']} AS INT)                AS age,
              {lower['customer_gender']}                          AS gender,
              CAST({lower['customer_satisfaction']} AS DOUBLE)    AS satisfaction
            FROM __raw_sales;
            """)
        else:
            _duck.execute(f"""
            CREATE OR REPLACE VIEW {DATA_TABLE} AS
            SELECT
              CAST(col0 AS DATE)      AS date,
              col1                    AS product,
              col2                    AS region,
              CAST(col3 AS DOUBLE)    AS sales,
              CAST(col4 AS INT)       AS age,
              col5                    AS gender,
              CAST(col6 AS DOUBLE)    AS satisfaction
            FROM read_csv_auto('{path_sql}', header=FALSE);
            """)

        cols = [r[1] for r in _duck.execute(f"PRAGMA table_info('{DATA_TABLE}')").fetchall()]
        _schema_cols_lower = {c.lower() for c in cols}
    return _duck

def wants_sql(user_q: str) -> Tuple[bool, str]:
    """
    Return (is_duckdb, reason). Word-aware checks to avoid false positives
    (e.g., 'sum' inside 'summarize'). Ordering:
      1) Explicit doc intent (unless clearly numeric/sales)
      2) Schema terms  -> duckdb
      3) Numeric cues  -> duckdb
      4) Sales terms   -> duckdb
      5) BI patterns   -> duckdb
      6) Default       -> docs
    """
    if not user_q or not user_q.strip():
        return False, "Empty question; default to docs."
    q = user_q.lower().strip()

    # 1) Doc intent wins unless it ALSO clearly asks for numeric/sales analysis
    if any(h in q for h in _DOC_HINTS):
        has_numeric = _any_word(q, _NUMERIC_WORDS) or any(p in q for p in _NUMERIC_PHRASES)
        has_sales   = _any_word(q, _SALES_WORDS)   or any(p in q for p in _SALES_PHRASES)
        if not (has_numeric or has_sales):
            return False, "Explicit doc intent detected."

    # YoY-by-quarter: always data
    if _has_yoy_quarter(q):
        return True, "Detected YoY-by-quarter sales intent → DuckDB"
    
    # 2) Schema terms → DuckDB
    schema_hits = [c for c in SALES_SCHEMA_COLUMNS if c in q]
    if schema_hits:
        return True, f"Schema terms detected: {', '.join(schema_hits)}"

    # 3) Numeric cues → DuckDB (word-aware OR phrase-aware)
    if _any_word(q, _NUMERIC_WORDS) or any(p in q for p in _NUMERIC_PHRASES):
        return True, "Numeric/analytic cues suggest table aggregation."

    # 4) Sales-domain terms → DuckDB (word-aware OR phrase-aware)
    if _any_word(q, _SALES_WORDS) or any(p in q for p in _SALES_PHRASES):
        return True, "Sales-domain terms suggest CSV source."

    # 5) BI phrasing → DuckDB (keep my _BI_PATTERNS)
    for pat in _BI_PATTERNS:
        if re.search(pat, q):
            return True, f"BI pattern matched: {pat}"

    # 6) Default to docs
    return False, "No strong CSV/analytic cues; route to docs."

def route_for_question(user_q: str) -> str:
    return "duckdb" if wants_sql(user_q)[0] else "docs"

def _is_safe_select(sql: str) -> Tuple[bool, str]:
    s = (sql or "").strip()
    if not SAFE_SELECT_RE.match(s):
        return False, "Only SELECT statements are allowed."
    for pat in FORBIDDEN_SQL_PATTERNS:
        if re.search(pat, s, re.IGNORECASE):
            return False, f"Forbidden token matched: {pat}"
    return True, "OK"

def _ensure_limit(sql: str, limit: int = 200) -> str:
    # If the query already has a LIMIT, leave it; else add one
    if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {limit}"

def _run_duckdb_sql(sql: str):
    con = _ensure_duckdb()
    cur = con.execute(sql)
    rows = cur.fetchall() or []
    headers = [c[0] for c in cur.description] if cur.description else []
    return {"headers": headers, "rows": rows}

def _run_duckdb_sql_retry(sql: str, *, table: str) -> Dict[str, Any]:
    """
    1) Sanitize to DuckDB dialect
    2) Try once
    3) On failure, try one guided retry with a hint comment (no-op for DuckDB but aids logs)
    Returns: {"headers":[...], "rows":[...], "sql":"<sanitized>", "attempt":"1|2"}
    """
    con = _ensure_duckdb()
    cleaned = _sanitize_duckdb_sql(sql, table=table)
    try:
        cur = con.execute(cleaned)
        rows = cur.fetchall() or []
        headers = [c[0] for c in cur.description] if cur.description else []
        return {"headers": headers, "rows": rows, "sql": cleaned, "attempt": "1"}
    except Exception as e1:
        # Guided retry (comment is harmless; should help when inspecting the executed SQL)
        hinted = f"{cleaned}\n-- RETRY after: {str(e1)[:200]}\n-- Tip: prefer INTERVAL and date_trunc() in DuckDB"
        cur = con.execute(hinted)
        rows = cur.fetchall() or []
        headers = [c[0] for c in cur.description] if cur.description else []
        return {"headers": headers, "rows": rows, "sql": cleaned, "attempt": "2"}

def _normalize_data_answer_payload(out: Dict[str, Any]) -> Dict[str, Any]:
    """
    For data-driven answers, remove doc-style citations and attach a data_sources hint.
    """
    # kill doc citation shape if present
    if "citations" in out:
        out["citations"] = []
    # add a friendly data_sources hint for the UI
    ds = out.get("data_sources", [])
    if "sales_data.csv" not in ds:
        out["data_sources"] = ds + ["sales_data.csv"]
    return out

# -- improve attempt
@lru_cache(maxsize=1)
def _distinct_values_map() -> dict:
    """Read distincts once (per process)."""
    con = _ensure_duckdb()
    out = {}
    for d in _DIM_CANDIDATES:
        c = _col(d)
        if not c:
            continue
        try:
            rows = con.execute(f"SELECT DISTINCT {c} FROM {DATA_TABLE} WHERE {c} IS NOT NULL").fetchall()
            values = [str(r[0]).strip() for r in rows if r and r[0] is not None]
            out[d] = sorted(set(values), key=lambda x: x.lower())
        except Exception:
            out[d] = []
    return out

def _find_metric_in_query(ql: str) -> str:
    if any(m in ql for m in _METRIC_SAT): return "satisfaction"
    if any(m in ql for m in _METRIC_SALES): return "sales"
    # default heuristic: analytics words → sales; otherwise None
    if any(w in ql for w in ("trend", "growth", "decline", "compare", "correlation", "change", "performance")):
        return "sales"
    return ""

def _metric_sql(metric: str) -> tuple[str, str]:
    """Return (expression_sql, agg) for the metric."""
    if metric == "satisfaction":
        c = _col("satisfaction","csat")
        return (f"CAST({c} AS DOUBLE)", "AVG")
    # default to sales
    c = _col("transaction_value","sales","amount","revenue")
    return (f"CAST({c} AS DOUBLE)", "SUM")

def _detect_timegrain(ql: str) -> str:
    for k, g in _TIME_GRAINS.items():
        if k in ql:
            return g
    # infer quarterly if 'quarter' appears
    if "quarter" in ql: return "quarter"
    if "month" in ql or "monthly" in ql: return "month"
    if "year" in ql or "annual" in ql: return "year"
    return ""  # let SQL choose per compare hint

def _detect_compare(ql: str) -> tuple[str,str] | tuple[(),()]:
    for phrase, val in _COMPARE_TOKENS.items():
        if phrase in ql:
            return val
    return ("","")

def _detect_dimensions_and_filters(ql: str) -> tuple[list[str], dict]:
    """Return (dimensions, filters) where filters is {dim: value} if value found in dictincts."""
    dims = []
    filters = {}
    distincts = _distinct_values_map()
    # which dims are mentioned?
    for d in _DIM_CANDIDATES:
        if d in ql:
            dims.append(d)
    # try to bind a value for any dim with distincts (like 'north' region)
    tokens = set([t.strip(",.?!") for t in ql.split()])
    for d, vals in distincts.items():
        for v in vals:
            vs = v.lower()
            # allow multi-word values
            if vs in ql:
                filters[d] = v
                if d not in dims:
                    dims.append(d)
                break
        else:
            # single-token containment check (fallback)
            if d not in filters:
                hits = [v for v in vals if v.lower() in tokens]
                if hits:
                    filters[d] = hits[0]
                    if d not in dims:
                        dims.append(d)
    return dims, filters

def _build_sql_from_intent(user_query: str) -> tuple[str, str]:
    """
    Build generalized SQL from the prompt. Returns (sql, reason).
    Covers: timegrain group-by, last/prev comparisons, last2 quarters, YoY, correlations.
    """
    ql = user_query.lower()
    metric = _find_metric_in_query(ql) or "sales"
    expr, agg = _metric_sql(metric)
    dims, filters = _detect_dimensions_and_filters(ql)
    timegrain = _detect_timegrain(ql)
    compare_grain, compare_kind = _detect_compare(ql)

    c_date = _col("date")
    if not c_date:
        raise ValueError("No date column detected in dataset.")
    time_expr = f"date_trunc('{timegrain or 'month'}', CAST({c_date} AS DATE))"  # default month if none

    # Correlation case
    if "correlation" in ql and ("satisfaction" in ql and any(x in ql for x in ("transaction", "value", "purchase", "sales"))):
        # corr(metricA, metricB)
        c_txn = _col("transaction_value","sales","amount","revenue")
        c_sat = _col("satisfaction","csat")
        sql = f"SELECT corr(CAST({c_txn} AS DOUBLE), CAST({c_sat} AS DOUBLE)) AS corr_coef FROM {DATA_TABLE};"
        return (sql, "correlation between transaction value and satisfaction")

    # Last 2 quarters
    if compare_grain == "quarter" and compare_kind == "last2":
        sql = f"""
        WITH q AS (
          SELECT date_trunc('quarter', CAST({c_date} AS DATE)) AS qtr
          FROM {DATA_TABLE}
          GROUP BY 1
          ORDER BY qtr DESC
          LIMIT 2
        ),
        agg AS (
          SELECT date_trunc('quarter', CAST({c_date} AS DATE)) AS qtr,
                 {', '.join(_col(d) + ' AS ' + d for d in dims) if dims else 'NULL::INT AS _'}
                 {',' if dims else ''} {agg}({expr}) AS value
          FROM {DATA_TABLE}
          WHERE date_trunc('quarter', CAST({c_date} AS DATE)) IN (SELECT qtr FROM q)
          {''.join(f" AND {_col(d)} = '{filters[d]}'" for d in filters)}
          GROUP BY 1{',' if dims else ''} {', '.join(str(i+2) for i,_ in enumerate(dims)) if dims else ''}
        )
        SELECT qtr AS period, {', '.join(dims) + ',' if dims else ''} value
        FROM agg
        ORDER BY qtr DESC {', ' + ', '.join(dims) if dims else ''};
        """.strip()
        return (sql, "two most recent quarters")

    # Last/previous quarter delta, optionally by a dimension=value (ie, region='North')
    if compare_grain == "quarter" and compare_kind in ("last", "previous"):
        # Compute current and previous automatically
        where_filters = "".join(f" AND {_col(d)} = '{filters[d]}'" for d in filters)
        sql = f"""
        WITH base AS (
          SELECT date_trunc('quarter', CAST({c_date} AS DATE)) AS qtr,
                 {agg}({expr}) AS val
          FROM {DATA_TABLE}
          WHERE 1=1 {where_filters}
          GROUP BY 1
        ),
        two AS (
          SELECT *
          FROM base
          ORDER BY qtr DESC
          LIMIT 2
        )
        SELECT
          (SELECT val FROM two ORDER BY qtr DESC LIMIT 1) AS current_qtr_value,
          (SELECT val FROM two ORDER BY qtr DESC OFFSET 1) AS prev_qtr_value,
          (SELECT current_qtr_value - prev_qtr_value)     AS delta;
        """.strip()
        return (sql, "quarter-over-quarter delta (current vs previous)")

    # YoY by quarter
    if compare_grain == "year" and compare_kind == "yoy" and "quarter" in ql:
        sql = f"""
        WITH base AS (
          SELECT CAST({c_date} AS DATE) AS d, {expr} AS v
          FROM {DATA_TABLE}
        ),
        agg AS (
          SELECT EXTRACT(YEAR FROM d)::INT AS year, EXTRACT(QUARTER FROM d)::INT AS quarter,
                 {agg}(v) AS total
          FROM base
          GROUP BY 1,2
        )
        SELECT a.year, a.quarter, a.total,
               (a.total - b.total) AS yoy_delta
        FROM agg a
        LEFT JOIN agg b ON b.quarter = a.quarter AND b.year = a.year - 1
        ORDER BY a.year, a.quarter;
        """.strip()
        return (sql, "year-over-year by quarter")

    # Generic timegrain group-by with optional dims/filters (trends, comparisons implied)
    sql = f"""
    SELECT
      {time_expr} AS period
      {',' if dims else ''}{', '.join(_col(d) + ' AS ' + d for d in dims)}
      , {agg}({expr}) AS value
    FROM {DATA_TABLE}
    WHERE 1=1 {''.join(f" AND {_col(d)} = '{filters[d]}'" for d in filters)}
    GROUP BY 1{',' if dims else ''} {', '.join(str(i+2) for i,_ in enumerate(dims)) if dims else ''}
    ORDER BY period ASC {', ' + ', '.join(dims) if dims else ''};
    """.strip()
    return (sql, f"{agg.lower()}({metric}) by {timegrain or 'month'}" + (f" over {', '.join(dims)}" if dims else ""))

# ===================== DuckDB SQL sanitization =====================
_SANITIZE_RULES = [
    # dates/time
    (r"\bGETDATE\(\)", "current_timestamp"),
    (r"\bNOW\(\)", "current_timestamp"),
    (r"\bCURRENT_TIMESTAMP\(\)", "current_timestamp"),
    # DATEADD(datepart, number, date) → date + INTERVAL 'number datepart'
    # Handle QUARTER/MONTH/DAY cases commonly emitted by LLMs
    (r"DATEADD\s*\(\s*quarter\s*,\s*([-+]?\d+)\s*,\s*([^)]+?)\)", r"\2 + INTERVAL '\1 quarter'"),
    (r"DATEADD\s*\(\s*month\s*,\s*([-+]?\d+)\s*,\s*([^)]+?)\)",   r"\2 + INTERVAL '\1 month'"),
    (r"DATEADD\s*\(\s*day\s*,\s*([-+]?\d+)\s*,\s*([^)]+?)\)",     r"\2 + INTERVAL '\1 day'"),
    # TOP N → LIMIT N
    (r"SELECT\s+TOP\s+(\d+)\s", r"SELECT "),
    (r"\bOFFSET\s+0\s+ROWS?\b", ""),  # clean leftovers
    # ISNULL(a,b) → coalesce(a,b)
    (r"\bISNULL\s*\(", "coalesce("),
    # IIF(cond, a, b) → CASE WHEN cond THEN a ELSE b END
    (r"\bIIF\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)", r"CASE WHEN \1 THEN \2 ELSE \3 END"),
    # CONVERT(date, expr) → CAST(expr AS DATE)
    (r"\bCONVERT\s*\(\s*date\s*,\s*([^)]+)\)", r"CAST(\1 AS DATE)"),
    # NVL → coalesce
    (r"\bNVL\s*\(", "coalesce("),
    # == or === → =
    (r"(?<![=!<>])==+(?!=)", "="),
]

_SELECT_ONLY_RE = re.compile(r"(?is)(?:with\s+.+?\)\s*)?\s*(select\s+.+)$")

def _sanitize_to_select_only(sql: Optional[str]) -> Optional[str]:
    if not sql:
        return None
    cand = sql.strip()
    # keep last statement if multiple
    if ";" in cand:
        cand = cand.rsplit(";", 1)[-1].strip() or cand
    m = _SELECT_ONLY_RE.search(cand)
    if m:
        return m.group(1).strip()
    # fallback: scan chunks from the end
    parts = re.split(r";\s*", sql)
    for chunk in reversed(parts):
        mm = _SELECT_ONLY_RE.search(chunk)
        if mm:
            return mm.group(1).strip()
    return None

def _sanitize_duckdb_sql(sql: str, *, table: str) -> str:
    """
    Convert common non-DuckDB idioms to DuckDB equivalents, conservatively.
    - DATEADD('quarter', N, X)  -> (CAST(X AS DATE) + INTERVAL '<N*3> months')
    - DATEADD('month',   N, X)  -> (CAST(X AS DATE) + INTERVAL '<N> months')
    - CURRENT_DATE() / current_date() -> current_date
    - NOW() / now() -> current_timestamp
    - Normalize date_trunc casing
    """
    s = sql

    # CURRENT_DATE() -> current_date
    s = re.sub(r"\bcurrent_date\s*\(\s*\)", "current_date", s, flags=re.IGNORECASE)

    # NOW() -> current_timestamp
    s = re.sub(r"\bnow\s*\(\s*\)", "current_timestamp", s, flags=re.IGNORECASE)

    # DATE_TRUNC(...) / date_trunc(...) – leave as date_trunc (DuckDB standard)
    s = re.sub(r"\bdate_trunc\b", "date_trunc", s, flags=re.IGNORECASE)

    # DATEADD('quarter', N, X)  OR DATEADD('month', N, X)
    def _dateadd_repl(m):
        unit = m.group(1).lower()      # 'quarter' or 'month'
        val = int(m.group(2))          # can be negative
        expr = m.group(3).strip()      # date expression
        # Quarters -> months multiplier
        months = val * (3 if unit.startswith("quarter") else 1)
        sign = "-" if months < 0 else "+"
        abs_months = abs(months)
        return f"(CAST({expr} AS DATE) {sign} INTERVAL '{abs_months} months')"

    s = re.sub(
        r"\bdateadd\s*\(\s*'?(quarter|month)'?\s*,\s*(-?\d+)\s*,\s*([^)]+?)\s*\)",
        _dateadd_repl,
        s,
        flags=re.IGNORECASE
    )

    # Remove T-SQL style backticks
    s = s.replace("`", "")

    # Ensure the right table name is used (model sometimes screws up aliases)
    # if it used FROM sales_data but DATA_TABLE=='sales', rewrite.
    if table != "sales_data":
        s = re.sub(r"\bFROM\s+sales_data\b", f"FROM {table}", s, flags=re.IGNORECASE)

    return s

# ===================== Health / Inspect =====================
@app.get("/debug/json-ok")
def json_ok():
    return {"ok": True, "answer": 42}

@app.get("/health")
def health():
    try:
        ver = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=2).json()
    except Exception as e:
        ver = {"error": str(e)}
    try:
        colls = [c.name for c in client.list_collections()]
    except Exception as e:
        colls = [f"error: {e}"]
    return {"status": "ok", "ollama": ver, "chroma_collections": colls}

@app.get("/data/inspect")
def data_inspect():
    con = _ensure_duckdb()
    resolved = _resolve_sales_csv()
    exists = os.path.exists(resolved)
    size_bytes = os.path.getsize(resolved) if exists else 0
    cols = [r[1] for r in con.execute(f"PRAGMA table_info('{DATA_TABLE}')").fetchall()]
    row_count = con.execute(f"SELECT COUNT(*) FROM {DATA_TABLE}").fetchone()[0]
    sample_rows = con.execute(f"SELECT * FROM {DATA_TABLE} LIMIT 3").fetchall()
    return {
        "sales_csv_resolved": resolved,
        "exists": exists,
        "size_bytes": size_bytes,
        "table": DATA_TABLE,
        "row_count": row_count,
        "columns": cols,
        "sample_rows": sample_rows,
    }

# ===================== KPIs (DuckDB) =====================
@app.get("/analytics/kpi")
def kpi():
    con = _ensure_duckdb()
    # total sales, avg satisfaction, top region, top product
    total_sales = con.execute(f"SELECT SUM(sales)::DOUBLE FROM {DATA_TABLE}").fetchone()[0]
    avg_sat = con.execute(f"SELECT AVG(satisfaction)::DOUBLE FROM {DATA_TABLE}").fetchone()[0]
    top_region = con.execute(f"""
        SELECT region FROM (
          SELECT region, SUM(sales)::DOUBLE AS s
          FROM {DATA_TABLE} GROUP BY region ORDER BY s DESC
        ) LIMIT 1
    """).fetchone()
    top_product = con.execute(f"""
        SELECT product FROM (
          SELECT product, SUM(sales)::DOUBLE AS s
          FROM {DATA_TABLE} GROUP BY product ORDER BY s DESC
        ) LIMIT 1
    """).fetchone()
    return {
        "total_sales": float(total_sales) if total_sales is not None else None,
        "avg_satisfaction": float(avg_sat) if avg_sat is not None else None,
        "top_region": top_region[0] if top_region else None,
        "top_product": top_product[0] if top_product else None,
    }

# ===================== BI endpoints (DuckDB SQL) =====================
@app.get("/bi/region-divergence")
def bi_region_divergence():
    con = _ensure_duckdb()
    sql = f"""
    WITH s AS (
      SELECT
        region,
        epoch(CAST(date AS TIMESTAMP)) AS x,
        sales::DOUBLE                 AS y_sales,
        satisfaction::DOUBLE          AS y_sat
      FROM {DATA_TABLE}
    )
    SELECT
      region,
      regr_slope(y_sales, x)       AS slope_sales,
      regr_slope(y_sat,   x)       AS slope_sat,
      COUNT(*)                      AS n
    FROM s
    GROUP BY region
    HAVING slope_sales > 0 AND slope_sat < 0
    ORDER BY slope_sales DESC;
    """
    rows = con.execute(sql).fetchall()
    return {
        "question": "Which regions have growing sales but declining satisfaction?",
        "rows": rows,
        "columns": ["region","slope_sales","slope_sat","n"],
        "source_table": DATA_TABLE,
    }

@app.get("/bi/top-products-under-30")
def bi_top_products_u30(limit: int = 2):
    con = _ensure_duckdb()
    sql = f"""
    SELECT
      product,
      SUM(sales)::DOUBLE AS total_sales,
      COUNT(*)           AS n
    FROM {DATA_TABLE}
    WHERE age < 30
    GROUP BY product
    ORDER BY total_sales DESC
    LIMIT {int(limit)};
    """
    rows = con.execute(sql).fetchall()
    return {
        "question": "What are the top products by sales for customers under 30?",
        "rows": rows,
        "columns": ["product","total_sales","n"],
        "source_table": DATA_TABLE,
    }

@app.get("/bi/region-trends")
def bi_region_trends(regions: str):
    con = _ensure_duckdb()
    region_list = [r.strip() for r in regions.split(",") if r.strip()]
    if not region_list:
        return {"regions": [], "rows": [], "columns": ["month","region","sales","satisfaction"], "source_table": DATA_TABLE}
    safe = ",".join("'" + r.replace("'", "''") + "'" for r in region_list)
    sql = f"""
    SELECT
      date_trunc('month', date)::DATE AS month,
      region,
      SUM(sales)::DOUBLE              AS sales,
      AVG(satisfaction)::DOUBLE       AS satisfaction
    FROM {DATA_TABLE}
    WHERE region IN ({safe})
    GROUP BY 1,2
    ORDER BY 1,2;
    """
    rows = con.execute(sql).fetchall()
    return {
        "regions": region_list,
        "rows": rows,
        "columns": ["month","region","sales","satisfaction"],
        "source_table": DATA_TABLE,
    }

# ===================== LLM SQL Generator (JSON) =====================

def _llm_generate_sql(query: str) -> Dict[str, Any]:
    """
    Ask the model for a JSON object with a single SELECT suitable for DuckDB.
    Contract:
      {"route":"duckdb","sql":"SELECT ...", "explanation":"<1-2 sentences>"}
    If not answerable from the table, it should respond {"route":"docs"}.
    """
    table = DATA_TABLE  # e.g., "sales"
    schema = (
        f"Table: {table}\n"
        "Columns:\n"
        "  - date (DATE)\n"
        "  - region (TEXT)\n"
        "  - product (TEXT)\n"
        "  - age (INT)\n"
        "  - gender (TEXT)\n"
        "  - sales (DOUBLE)\n"
        "  - satisfaction (DOUBLE)\n"
        "  - transaction_value (DOUBLE, optional)\n"
    )
    sys = (
        f"You generate SQL for DuckDB over one table `{table}` only.\n"
        "Output STRICT JSON in one line with keys: route, sql, explanation.\n"
        f"If solvable from `{table}`, set route to \"duckdb\" and provide a single SELECT.\n"
        "If not solvable from table, set route to \"docs\" and omit sql.\n"
        "Rules:\n"
        "- SELECT only. No DDL/DML.\n"
        "- Use DuckDB syntax: use INTERVAL arithmetic (e.g., CURRENT_DATE - INTERVAL '3 months'), "
        "  date_trunc('quarter', <date>), lower(<text>).\n"
        "- Do NOT use DATEADD/DATE_SUB/EXTRACT functions; instead use INTERVAL arithmetic and date_trunc.\n"
        "- Keep it short and efficient."
    )
    prompt = (
        f"SYSTEM:\n{sys}\n\n"
        f"SCHEMA:\n{schema}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Return JSON ONLY. Example:\n"
        f"{{\"route\":\"duckdb\",\"sql\":\"SELECT COUNT(*) AS n FROM {table}\",\"explanation\":\"count rows\"}}"
    )

    def _call():
        return requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": os.environ.get("CHAT_MODEL", "phi3:mini"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1536")),
                    "num_predict": 160,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            },
            timeout=int(os.getenv("API_GENERATE_TIMEOUT", "45")) + 5,
        )

    resp = asyncio.run(asyncio.wait_for(run_in_threadpool(_call), timeout=int(os.getenv("API_GENERATE_TIMEOUT", "45"))))
    resp.raise_for_status()
    raw = (resp.json() or {}).get("response", "").strip()

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
    except Exception as e:
        return {"route": "docs", "error": f"Bad JSON from model: {e}", "raw": raw}

    if not isinstance(obj, dict) or "route" not in obj:
        return {"route": "docs", "error": "No 'route' in JSON.", "raw": raw}
    return obj

# ===================== Time series (history) =====================
@app.get("/ts/sales-daily")
def ts_sales_daily():
    con = _ensure_duckdb()
    rows = con.execute(f"""
        SELECT
          CAST(date AS DATE) AS date,
          SUM(sales)::DOUBLE AS sales
        FROM {DATA_TABLE}
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    return {
        "columns": ["date", "sales"],
        "rows": rows,
        "source_table": DATA_TABLE,
        "n": len(rows),
    }

# ===================== Forecast helpers =====================
def _compute_forecast_from_hist(
    hist: List[Tuple[datetime, float]],
    h: int,
    algo: str,
    window: int
) -> List[List[object]]:
    """
    hist: list of (date, sales) sorted by date ascending
    Returns: list of [date_str, forecast_value]
    """
    if not hist:
        return []
    last_date = hist[-1][0]
    values = [float(v) for (_, v) in hist]

    h = max(1, min(int(h), 365))
    window = max(1, min(int(window), len(values)))

    fcst: List[List[object]] = []

    if algo == "seasonal7":
        if len(values) < 7:
            raise HTTPException(status_code=400, detail="Need >= 7 history points for seasonal7")
        buf = values[:]  # rolling buffer
        cur = last_date
        for _ in range(h):
            cur = cur + timedelta(days=1)
            next_val = buf[-7]  # repeat value from t-7
            buf.append(next_val)
            fcst.append([_to_datestr(cur), float(next_val)])

    elif algo == "drift":
        if len(values) < 2:
            raise HTTPException(status_code=400, detail="Need >= 2 history points for drift")
        w = min(window, len(values))
        y0 = float(values[-w])
        yT = float(values[-1])
        T = w - 1 if w > 1 else 1
        slope = (yT - y0) / T
        cur = last_date
        for i in range(1, h + 1):
            cur = cur + timedelta(days=1)
            val = yT + slope * i
            fcst.append([_to_datestr(cur), float(val)])

    else:  # ma7_baseline (flat)
        w = min(window, len(values))
        base = sum(values[-w:]) / float(w)
        cur = last_date
        for _ in range(h):
            cur = cur + timedelta(days=1)
            fcst.append([_to_datestr(cur), float(base)])

    return fcst

def _load_hist(con) -> List[Tuple[datetime, float]]:
    rows = con.execute(f"""
        SELECT CAST(date AS DATE) AS date, SUM(sales)::DOUBLE AS sales
        FROM {DATA_TABLE}
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    return [(r[0], float(r[1])) for r in rows]

# ===================== Forecast endpoints =====================
@app.get("/api/ts-forecast-v2")
def ts_forecast_v2(
    h: int = Query(30, ge=1, le=365),
    algo: str = Query("ma7_baseline"),
    window: int = Query(7, ge=1, le=60)
):
    """
    Final forecast endpoint.
    algos:
      - ma7_baseline (default): flat mean of last `window` days
      - seasonal7: repeats value from 7 days ago (weekday seasonality)
      - drift: naive with linear drift from last `window` days
    """
    algo = (algo or "ma7_baseline").lower()
    con = _ensure_duckdb()
    hist = _load_hist(con)
    if not hist:
        return {
            "model": algo,
            "history_columns": ["date","sales"],
            "history": [],
            "forecast_columns": ["date","sales_hat"],
            "forecast": [],
        }

    fcst = _compute_forecast_from_hist(hist, h=h, algo=algo, window=window)
    history = [[_to_datestr(d), float(v)] for (d, v) in hist]

    return {
        "model": algo,
        "history_columns": ["date", "sales"],
        "history": history,
        "forecast_columns": ["date", "sales_hat"],
        "forecast": fcst,
    }

@app.get("/api/ts-forecast")
def ts_forecast_api_legacy(h: int = 30, algo: str = "ma7_baseline", window: int = 7):
    """
    DEPRECATED: thin wrapper to v2 for backward-compatibility.
    """
    return ts_forecast_v2(h=h, algo=algo, window=window)

# ===================== RAG (unchanged from your working version) =====================
from ollama_embedder import OllamaEmbeddingFunction
_embedder = OllamaEmbeddingFunction()

SYSTEM_PROMPT = (
    "You are a concise BI analyst. Use ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Answer in <=120 words."
)
RAG_K                 = int(os.getenv("RAG_K", "3"))
RAG_MAX_INPUT_CHARS   = int(os.getenv("RAG_MAX_INPUT_CHARS", "3500"))
RAG_NUM_PREDICT       = int(os.getenv("RAG_NUM_PREDICT", "160"))
RAG_TEMPERATURE       = float(os.getenv("RAG_TEMPERATURE", "0.2"))
RAG_TOP_P             = float(os.getenv("RAG_TOP_P", "0.9"))
API_GENERATE_TIMEOUT  = int(os.getenv("API_GENERATE_TIMEOUT", "45"))
OLLAMA_NUM_CTX        = int(os.getenv("OLLAMA_NUM_CTX", "1536"))

def _truncate_chars(s: str, limit: int) -> str:
    if len(s) <= limit: return s
    cut = s[:limit]; last_para = cut.rfind("\n\n")
    return cut[: last_para if last_para > 400 else limit]

async def _with_timeout(coro, seconds: int):
    return await asyncio.wait_for(coro, timeout=seconds)

def _rows_to_markdown(rows: list, headers: list, max_rows: int = 8) -> str:
    if not rows: return ""
    hdr = "| " + " | ".join(headers) + " |\n"
    sep = "| " + " | ".join("---" for _ in headers) + " |\n"
    body = ""
    for r in rows[:max_rows]:
        body += "| " + " | ".join("" if v is None else str(v) for v in r) + " |\n"
    return hdr + sep + body

def _col(*cands: str) -> str:
    for cand in cands:
        cl = cand.lower()
        for have in _schema_cols_lower:
            if have == cl:
                return have
    synonyms = {
        "date": ["date", "dt", "day", "order_date", "week", "week_start", "week_ending"],
        "region": ["region", "state", "area", "market"],
        "product": ["product", "product_name", "sku", "item"],
        "sales": ["sales", "revenue", "amount", "total_sales", "weekly_sales"],
        "satisfaction": ["satisfaction", "csat", "nps", "customer_satisfaction"],
        "age": ["age", "customer_age", "age_years"],
    }
    for cand in cands:
        key = cand.lower()
        if key in synonyms:
            for alt in synonyms[key]:
                if alt in _schema_cols_lower:
                    return alt
    return ""

def _answer_regions_growth_vs_csat():
    con = _ensure_duckdb()
    c_date = _col("date", "week")
    c_region = _col("region")
    c_sales = _col("sales", "weekly_sales")
    c_sat = _col("satisfaction", "csat")
    if not all([c_date, c_region, c_sales, c_sat]):
        return None
    q = f"""
    WITH base AS (
      SELECT
        {c_region} AS region,
        CAST({c_sales} AS DOUBLE) AS sales,
        CAST({c_sat}   AS DOUBLE) AS sat,
        ROW_NUMBER() OVER (PARTITION BY {c_region} ORDER BY {c_date})::DOUBLE AS t
      FROM {DATA_TABLE}
      WHERE {c_sales} IS NOT NULL AND {c_sat} IS NOT NULL
    ),
    reg AS (
      SELECT
        region,
        regr_slope(sales, t) AS slope_sales,
        regr_slope(sat,   t) AS slope_sat,
        COUNT(*) AS n
      FROM base
      GROUP BY region
    )
    SELECT region, round(slope_sales,6) AS slope_sales, round(slope_sat,6) AS slope_sat, n
    FROM reg
    WHERE slope_sales > 0 AND slope_sat < 0
    ORDER BY slope_sales DESC, slope_sat ASC
    LIMIT 10;
    """
    rows = con.execute(q).fetchall()
    if rows is None: rows = []
    return {"headers": ["region", "slope_sales", "slope_sat", "n"], "rows": rows}

def _answer_top_products_under_30():
    con = _ensure_duckdb()
    c_prod  = _col("product", "product_name", "sku", "item")
    c_sales = _col("sales", "weekly_sales")
    c_age   = _col("age", "customer_age")
    if not all([c_prod, c_sales, c_age]):
        return None
    q = f"""
    SELECT {c_prod} AS product, SUM(CAST({c_sales} AS DOUBLE)) AS total_sales
    FROM {DATA_TABLE}
    WHERE CAST({c_age} AS DOUBLE) < 30
    GROUP BY product
    ORDER BY total_sales DESC
    LIMIT 2;
    """
    rows = con.execute(q).fetchall()
    if rows is None: rows = []
    return {"headers": ["product", "total_sales"], "rows": rows}

def _answer_month_with_highest_sales_growth():
    con = _ensure_duckdb()
    c_date = _col("date","week")
    c_sales = _col("sales","weekly_sales")
    if not all([c_date, c_sales]): return None
    q = f"""
      WITH d AS (
        SELECT CAST({c_date} AS DATE) AS d, SUM(CAST({c_sales} AS DOUBLE)) AS sales
        FROM {DATA_TABLE} GROUP BY 1
      ),
      m AS (
        SELECT date_trunc('month', d) AS month, SUM(sales) AS m_sales
        FROM d GROUP BY 1
      ),
      g AS (
        SELECT month, m_sales,
               m_sales - LAG(m_sales) OVER (ORDER BY month) AS mom_growth
        FROM m
      )
      SELECT month, m_sales, mom_growth
      FROM g ORDER BY mom_growth DESC NULLS LAST LIMIT 1;
    """
    rows = con.execute(q).fetchall() or []
    return {"headers": ["month","m_sales","mom_growth"], "rows": rows}

def _answer_gender_vs_avg_satisfaction():
    con = _ensure_duckdb()
    c_gender = _col("gender")
    c_sat = _col("satisfaction","csat")
    if not all([c_gender, c_sat]): return None
    q = f"""
      SELECT {c_gender} AS gender, AVG(CAST({c_sat} AS DOUBLE)) AS avg_satisfaction
      FROM {DATA_TABLE}
      WHERE {c_sat} IS NOT NULL AND {c_gender} IS NOT NULL
      GROUP BY gender ORDER BY avg_satisfaction DESC;
    """
    rows = con.execute(q).fetchall() or []
    return {"headers": ["gender","avg_satisfaction"], "rows": rows}

def _answer_satisfaction_change_region_last_quarter(user_query: str):
    """
    Computes previous quarter vs current quarter average satisfaction for a given region,
    where 'current quarter' is derived from MAX(date) in the dataset.
    Returns a small table with avg_prev, avg_curr, and delta.
    """
    ql = user_query.lower()
    regions = ["north","south","east","west"]
    region = next((r for r in regions if r in ql), None)
    if not region:
        return None  # only handle explicit region asks

    con = _ensure_duckdb()
    c_date = _col("date")
    c_region = _col("region")
    c_sat = _col("satisfaction", "csat")

    if not all([c_date, c_region, c_sat]):
        return None

    q = f"""
    WITH mx AS (
      SELECT max(CAST({c_date} AS DATE)) AS maxd FROM {DATA_TABLE}
    ),
    b AS (
      SELECT
        date_trunc('quarter', maxd)                          AS q_curr_start,
        date_trunc('quarter', maxd) - INTERVAL '3 months'    AS q_prev_start
      FROM mx
    ),
    curr AS (
      SELECT AVG(CAST({c_sat} AS DOUBLE)) AS avg_curr
      FROM {DATA_TABLE}, b
      WHERE lower({c_region}) = '{region.lower()}'
        AND CAST({c_date} AS DATE) >= b.q_curr_start
        AND CAST({c_date} AS DATE) <  b.q_curr_start + INTERVAL '3 months'
    ),
    prev AS (
      SELECT AVG(CAST({c_sat} AS DOUBLE)) AS avg_prev
      FROM {DATA_TABLE}, b
      WHERE lower({c_region}) = '{region.lower()}'
        AND CAST({c_date} AS DATE) >= b.q_prev_start
        AND CAST({c_date} AS DATE) <  b.q_curr_start
    )
    SELECT initcap('{region.lower()}') AS region, avg_prev, avg_curr, (avg_curr - avg_prev) AS delta;
    """
    rows = con.execute(q).fetchall() or []
    if not rows:
        return None
    return {
        "headers": ["region","avg_prev","avg_curr","delta"],
        "rows": rows
    }

def _answer_avg_satisfaction_by_region_two_quarters():
    """
    Returns avg satisfaction by region for the two most recent quarters present in the data.
    """
    con = _ensure_duckdb()
    c_date = _col("date")
    c_region = _col("region")
    c_sat = _col("satisfaction","csat")
    if not all([c_date, c_region, c_sat]): 
        return None

    q = f"""
    WITH q AS (
      SELECT date_trunc('quarter', CAST({c_date} AS DATE)) AS qtr
      FROM {DATA_TABLE}
      GROUP BY 1
      ORDER BY qtr DESC
      LIMIT 2
    ),
    agg AS (
      SELECT 
        date_trunc('quarter', CAST({c_date} AS DATE)) AS qtr,
        {c_region} AS region,
        AVG(CAST({c_sat} AS DOUBLE)) AS avg_sat
      FROM {DATA_TABLE}
      WHERE date_trunc('quarter', CAST({c_date} AS DATE)) IN (SELECT qtr FROM q)
      GROUP BY 1,2
    )
    SELECT qtr::DATE AS quarter_start, region, avg_sat
    FROM agg
    ORDER BY quarter_start DESC, region ASC;
    """
    rows = con.execute(q).fetchall() or []
    return {"headers": ["quarter_start","region","avg_sat"], "rows": rows}

def _any_word(q: str, words: Tuple[str, ...]) -> bool:
    # match whole words only: \bword\b
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)

# split into single words vs phrases
_NUMERIC_WORDS: Tuple[str, ...] = (
    "sum","avg","average","median","min","max","total",
    "top","rank","trend","increase","decrease","growth","decline",
    "yoy","mom","qoq","quarter","month","weekly","highest","lowest",
    "compare","correlation","corr","distribution","bucket","percentile",
    "quartile","std","variance",
)
_NUMERIC_PHRASES: Tuple[str, ...] = (
    "y/y","m/m","q/q","by region","by product","by age","by gender",
    "segment","breakdown",
)

_SALES_WORDS: Tuple[str, ...] = (
    "sales","revenue","txn","transaction","customers","customer",
    "satisfaction","nps","age","gender","region","product",
)
_SALES_PHRASES: Tuple[str, ...] = ("transaction value",)


def _summarize_table_with_llm(user_query: str, table_dict: Dict[str, Any], source: str) -> Dict[str, Any]:
    md = _rows_to_markdown(table_dict["rows"], table_dict["headers"])
    prompt = (
        f"SYSTEM:\nYou are a concise BI analyst. Use ONLY the provided context. If the answer is not in the context, say you don't know. Answer in <=120 words.\n\n"
        f"USER QUESTION:\n{user_query}\n\n"
        "CONTEXT (tabular result derived directly from the dataset):\n"
        f"{md}\n\n"
        "Reply with one concise paragraph (<=120 words) that answers the question strictly from the table.\n"
        f"Then write:\nSources:\n- {source}\n"
    )
    chat_model = os.environ.get("CHAT_MODEL", "phi3:mini")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": chat_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1536")),
                    "num_predict": int(os.getenv("RAG_NUM_PREDICT", "160")),
                    "temperature": float(os.getenv("RAG_TEMPERATURE", "0.2")),
                    "top_p": float(os.getenv("RAG_TOP_P", "0.9")),
                },
            },
            timeout=int(os.getenv("API_GENERATE_TIMEOUT", "45")) + 5,
        )
        resp.raise_for_status()
        out = resp.json().get("response", "").strip()
    except Exception as e:
        out = f"(Model error while summarizing table: {e})\n\n{md}\n\nSources:\n- {source}"
    return {
        "answer": out,
        "citations": [{"index": 1, "source": source, "page": None, "chunk": None}],
        "table": {"headers": table_dict["headers"], "rows": table_dict["rows"]},
    }

def _maybe_answer_with_data(user_query: str):
    ql = user_query.lower()
    if ("satisfaction" in ql) and ("last quarter" in ql) and any(r in ql for r in ["north","south","east","west"]):
        tbl = _answer_satisfaction_change_region_last_quarter(user_query)
        if isinstance(tbl, dict) and tbl.get("rows"):
            return _summarize_table_with_llm(user_query, tbl, source="sales_data.csv")
    if (("region" in ql) or ("regions" in ql)) and ("grow" in ql or "increas" in ql) and ("satisfaction" in ql or "csat" in ql):
        tbl = _answer_regions_growth_vs_csat()
        if isinstance(tbl, dict) and tbl.get("rows"):
            return _summarize_table_with_llm(user_query, tbl, source="sales_data.csv")
    if (("top" in ql) or ("best" in ql)) and ("product" in ql) and (("under 30" in ql) or ("< 30" in ql) or ("younger than 30" in ql)):
        tbl = _answer_top_products_under_30()
        if isinstance(tbl, dict) and tbl.get("rows"):
            return _summarize_table_with_llm(user_query, tbl, source="sales_data.csv")
    if "month" in ql and ("highest" in ql or "largest" in ql) and ("growth" in ql or "increase" in ql) and "sales" in ql:
        tbl = _answer_month_with_highest_sales_growth()
        if isinstance(tbl, dict) and tbl.get("rows"):
            return _summarize_table_with_llm(user_query, tbl, source="sales_data.csv")
    if ("gender" in ql) and ("satisfaction" in ql or "csat" in ql or "avg" in ql or "average" in ql):
        tbl = _answer_gender_vs_avg_satisfaction()
        if isinstance(tbl, dict) and tbl.get("rows"):
            return _summarize_table_with_llm(user_query, tbl, source="sales_data.csv")
    if ("two most recent quarter" in ql or "last two quarter" in ql) and \
       ("avg" in ql or "average" in ql) and \
       ("satisfaction" in ql) and ("region" in ql):
        tbl = _answer_avg_satisfaction_by_region_two_quarters()
        if isinstance(tbl, dict) and tbl.get("rows"):
            return _summarize_table_with_llm(user_query, tbl, source="sales_data.csv")
    return None

@app.post("/rag/query")
def rag_query(payload: Dict[str, Any] = Body(...)):
    query: str = payload.get("query", "")
    k_env = int(os.getenv("RAG_K", "3"))
    k: int = int(payload.get("k", k_env))

    if not query.strip():
        return {"answer": "Please provide a question.", "citations": [], "source_used": None}

    # ── Deterministic routing gate ────────────────────────────────────
    decision = "docs"
    route_reason = "router unavailable"
    try:
        is_duckdb, route_reason = wants_sql(query)
        decision = "duckdb" if is_duckdb else "docs"
    except Exception:
        decision = "docs"
        route_reason = "router error; defaulted to docs"

    # ── DATA PATH ─────────────────────────────────────────────────────
    if decision == "duckdb":
        # 1) Fast data-aware templates (your existing function)
        try:
            data_ans = _maybe_answer_with_data(query)
            if data_ans:
                if "source_used" not in data_ans:
                    data_ans["source_used"] = "duckdb"
                data_ans["route_reason"] = route_reason + "; matched data template"
                return _normalize_data_answer_payload(data_ans)
        except HTTPException:
            raise
        except Exception:
            # swallow and try generalized builder next
            pass

        # 2) Generalized intent → SQL builder (schema-aware)
        try:
            gen_sql, gen_reason = _build_sql_from_intent(query)  # ← added in previous step
            # Safety + execution with repair+retry
            gen_sql = _sanitize_duckdb_sql(gen_sql, table=DATA_TABLE)
            ok, msg = _is_safe_select(gen_sql)
            if ok:
                gen_sql = _ensure_limit(gen_sql, limit=int(os.getenv("ASK_AI_SQL_LIMIT", "200")))
                tbl = _run_duckdb_sql_retry(gen_sql, table=DATA_TABLE)   # ← repair+retry
                if tbl and tbl.get("rows"):
                    out = _summarize_table_with_llm(query, {"headers": tbl["headers"], "rows": tbl["rows"]}, source="sales_data.csv")
                    out["source_used"]  = "duckdb"
                    out["sql"]          = tbl.get("sql", gen_sql)
                    out["route_reason"] = f"{route_reason}; intent→SQL: {gen_reason}"
                    return _normalize_data_answer_payload(out)
            else:
                route_reason = f"{route_reason}; intent SQL unsafe ({msg})"
        except NameError:
            # _build_sql_from_intent not present: skip to LLM-SQL
            pass
        except Exception as e:
            # Builder failed; continue to LLM-SQL
            route_reason = f"{route_reason}; intent builder error: {e}"

        # 3) LLM-SQL fallback (SELECT-only clamp + limit + repair+retry)
        try:
            plan = _llm_generate_sql(query)  # {"route":"duckdb"|"docs", "sql": "...", ...}
            if plan.get("route") == "duckdb" and plan.get("sql"):
                sel = _sanitize_to_select_only(plan.get("sql"))
                if sel:
                    sql = _sanitize_duckdb_sql(sel, table=DATA_TABLE)
                    ok, msg = _is_safe_select(sql)
                    if ok:
                        sql = _ensure_limit(sql, limit=int(os.getenv("ASK_AI_SQL_LIMIT", "200")))
                        tbl = _run_duckdb_sql_retry(sql, table=DATA_TABLE)  # repair+retry
                        if tbl and tbl.get("rows"):
                            summarized = _summarize_table_with_llm(
                                query,
                                {"headers": tbl["headers"], "rows": tbl["rows"]},
                                source="sales_data.csv"
                            )
                            summarized["source_used"]  = "duckdb"
                            summarized["sql"]          = tbl.get("sql", sql)
                            summarized["route_reason"] = f"{route_reason}; LLM-SQL ({'retry' if tbl.get('attempt')=='2' else 'first try'})"
                            return _normalize_data_answer_payload(summarized)
                    else:
                        route_reason = f"{route_reason}; LLM-SQL unsafe ({msg})"
                else:
                    route_reason = f"{route_reason}; LLM-SQL contained no usable SELECT"
            else:
                route_reason = f"{route_reason}; LLM suggested docs"
        except Exception as e:
            route_reason = f"{route_reason}; LLM-SQL error: {e}"

    # If we got here and the decision was DuckDB, do NOT fall back to docs.
    if decision == "duckdb":
        return _normalize_data_answer_payload({
            "answer": "I couldn’t compute a confident answer from the dataset for that question.",
            "citations": [],
            "data_sources": ["sales_data.csv"],
            "source_used": "duckdb",
            "route_reason": route_reason,
            "sql": "",
            "table": {"headers": [], "rows": []},
        })

    # ── DOC RAG path (unchanged logic, with source annotations) ───────
    coll = get_docs_collection()
    try:
        from ollama_embedder import OllamaEmbeddingFunction
        _embedder = OllamaEmbeddingFunction()
        qvec = _embedder.embed_one(query)
    except Exception as e:
        return {
            "answer": f"(Embedding error: {e})",
            "citations": [],
            "source_used": "docs",
            "route_reason": route_reason,
        }

    n = max(1, min(k, 10))  # Keep n_results sane
    res = coll.query(query_embeddings=[qvec], n_results=n)

    docs  = (res.get("documents")  or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    if not docs:
        return {
            "answer": "I couldn't find anything relevant in the indexed documents.",
            "citations": [],
            "source_used": "docs",
            "route_reason": route_reason,
        }

    context_lines: List[str] = []
    citations = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        excerpt = " ".join((d or "").split())
        excerpt = textwrap.shorten(excerpt, width=700, placeholder=" …")
        context_lines.append(f"[{i}] {excerpt}")
        citations.append({
            "index": i,
            "source": m.get("source"),
            "page": m.get("page"),
            "chunk": m.get("chunk"),
        })

    def _call_ollama():
        return requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": os.environ.get("CHAT_MODEL", "phi3:mini"),
                "prompt": (
                    "SYSTEM:\nYou are a concise BI analyst. Use ONLY the provided context. "
                    "If the answer is not in the context, say you don't know. Answer in <=120 words.\n\n"
                    f"USER QUESTION:\n{query}\n\n"
                    f"CONTEXT (top {n} snippets):\n" + "\n".join(context_lines)
                    + "\n\nReply with one paragraph (<=120 words). Then write:\nSources: [Doc Name p.X] style."
                ),
                "stream": False,
                "options": {
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1536")),
                    "num_predict": int(os.getenv("RAG_NUM_PREDICT", "160")),
                    "temperature": float(os.getenv("RAG_TEMPERATURE", "0.2")),
                    "top_p": float(os.getenv("RAG_TOP_P", "0.9")),
                },
            },
            timeout=int(os.getenv("API_GENERATE_TIMEOUT", "45")) + 5,
        )

    try:
        resp = asyncio.run(
            asyncio.wait_for(
                run_in_threadpool(_call_ollama),
                timeout=int(os.getenv("API_GENERATE_TIMEOUT", "45"))
            )
        )
        resp.raise_for_status()
        out = resp.json().get("response", "").strip()
    except asyncio.TimeoutError:
        best = context_lines[0] if context_lines else "No context found."
        return {
            "answer": best + "\n\n_Sources shown below — model timed out; showing top snippet instead._",
            "citations": citations,
            "source_used": "docs",
            "route_reason": route_reason,
        }
    except Exception as e:
        out = f"(Model error: {e})"

    if citations:
        src_lines = [f"[{c['index']}] {c['source']} p.{c['page']} c.{c['chunk']}" for c in citations]
        out = f"{out}\n\nSources:\n" + "\n".join(src_lines)

    return {
        "answer": out,
        "citations": citations,
        "source_used": "docs",
        "route_reason": route_reason,
    }

# ===================== RAG stats =====================
@app.get("/rag/stats")
def rag_stats():
    try:
        col = get_docs_collection()
        res = col.get(limit=1)
        sample_ids = res.get("ids")
        return {"collection": "docs", "ok": True, "sample_ids": sample_ids}
    except Exception as e:
        return {"collection": "docs", "ok": False, "error": str(e)}
