import os
import requests
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Intellect BI", layout="wide")

# Altair safety (large datasets)
alt.data_transformers.disable_max_rows()

# Center everything in the sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            text-align: center;
        }
        [data-testid="stSidebar"] .element-container {
            display: flex;
            justify-content: center;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

# add my cool logo
logo = Image.open("intellect-bi-logo.png")
st.sidebar.image(logo, width=200)
st.sidebar.caption("Containerized BI + Local AI")

tabs = st.tabs(["Overview", "Drill-down", "Insights", "Forecast", "Anomalies", "Ask AI", "System"])

# -------------------- Overview --------------------
with tabs[0]:
    st.header("Overview")
    try:
        r = requests.get(f"{API_URL}/analytics/kpi", timeout=10)
        r.raise_for_status()
        k = r.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{k.get('total_sales', 0):,.0f}")
        c2.metric("Avg Satisfaction", f"{k.get('avg_satisfaction', 0):.2f}")
        c3.metric("Top Region", k.get("top_region", "—"))
        c4.metric("Top Product", k.get("top_product", "—"))
    except Exception as e:
        st.error(f"Failed to load KPIs: {e}")

# -------------------- Drill-down --------------------
with tabs[1]:
    st.header("Drill-down")
    st.write("Region / Product breakdowns.")

# -------------------- Insights --------------------
with tabs[2]:
    st.header("Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Growing Sales, Declining Satisfaction (by Region)")
        try:
            r = requests.get(f"{API_URL}/bi/region-divergence", timeout=15)
            r.raise_for_status()
            data = r.json()
            div_rows = data.get("rows", [])
            if div_rows:
                df_div = pd.DataFrame(div_rows, columns=data.get("columns", None))
                st.dataframe(df_div, use_container_width=True)
                regions = df_div["region"].tolist()

                # Fetch monthly trends for the regions
                rt = requests.get(
                    f"{API_URL}/bi/region-trends",
                    params={"regions": ",".join(regions)},
                    timeout=20,
                )
                rt.raise_for_status()
                trend_payload = rt.json()
                trend = trend_payload.get("rows", [])
                cols = trend_payload.get("columns", ["month", "region", "sales", "satisfaction"])
                if trend:
                    df_trend = pd.DataFrame(trend, columns=cols)
                    # Sales by month (pivot to wide for line chart)
                    sales_pivot = df_trend.pivot(index="month", columns="region", values="sales").sort_index()
                    st.caption("Monthly Sales by Region (for diverging regions)")
                    st.line_chart(sales_pivot)

                    sat_pivot = df_trend.pivot(index="month", columns="region", values="satisfaction").sort_index()
                    st.caption("Monthly Avg Satisfaction by Region (for diverging regions)")
                    st.line_chart(sat_pivot)
                else:
                    st.info("No monthly trend rows returned for the selected regions.")
            else:
                st.info("No regions currently show ↑ sales and ↓ satisfaction.")
        except Exception as e:
            st.error(f"Error fetching region divergence: {e}")

    with col2:
        st.subheader("Top Products by Sales (Customers Under 30)")
        try:
            r = requests.get(f"{API_URL}/bi/top-products-under-30", params={"limit": 2}, timeout=15)
            r.raise_for_status()
            data = r.json()
            rows = data.get("rows", [])
            cols = data.get("columns", ["product", "total_sales", "n"])
            if rows:
                df = pd.DataFrame(rows, columns=cols)
                st.dataframe(df, use_container_width=True)
                # Bar chart
                bar = df.set_index("product")["total_sales"]
                st.bar_chart(bar)
            else:
                st.info("No results.")
        except Exception as e:
            st.error(f"Error fetching top products: {e}")

# -------------------- Forecast --------------------
with tabs[3]:
    st.header("Forecast")

    left, right = st.columns([3, 1])
    with left:
        horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=120, value=30, step=1)
    with right:
        show_recent = st.checkbox("Show recent only", value=False, help="Show only last 90 days of history + forecast")

    # Forecast controls
    algo = st.selectbox(
        "Forecast algorithm",
        ["ma7_baseline", "seasonal7", "drift"],
        index=0,
        help="ma7_baseline: flat mean of last N days · seasonal7: repeats value from 7 days ago · drift: linear trend from last N days",
    )
    window = st.number_input(
        "Window (days)", min_value=2, max_value=60, value=7, step=1,
        help="How many trailing days to use for baseline/drift calculations."
    )

    try:
        # --- API call ---
        params = {"h": int(horizon), "algo": algo}
        if algo in ("ma7_baseline", "drift"):
            params["window"] = int(window)

        r = requests.get(f"{API_URL}/api/ts-forecast-v2", params=params, timeout=20)
        r.raise_for_status()
        js = r.json()  # keep `js` name consistent throughout

        # ---------- helpers ----------
        def parse_block(js_obj, data_key, cols_key):
            rows = js_obj.get(data_key, []) or []
            cols = [str(c).strip().lower() for c in (js_obj.get(cols_key, []) or [])]
            if not rows:
                return pd.DataFrame(columns=["date", "value"])

            if isinstance(rows[0], (list, tuple)) and cols:
                df = pd.DataFrame(rows, columns=cols)
            elif isinstance(rows[0], dict):
                df = pd.DataFrame(rows)
                df.columns = [str(c).strip().lower() for c in df.columns]
                if cols:
                    lowmap = {str(c).lower(): c for c in df.columns}
                    rename_map = {lowmap[c]: c for c in cols if c in lowmap}
                    if rename_map:
                        df = df.rename(columns=rename_map)
            else:
                df = pd.DataFrame(rows)

            df.columns = [str(c).strip().lower() for c in df.columns]
            return df

        def coerce_series(df_raw: pd.DataFrame, prefer_values=("sales", "value")) -> pd.DataFrame:
            cols = [c.lower() for c in df_raw.columns]
            df = df_raw.copy()
            if "date" not in cols and "ds" in cols:
                df = df.rename(columns={"ds": "date"})
            if "date" not in [c.lower() for c in df.columns]:
                raise ValueError(f"Missing 'date' column. Got {list(df.columns)}")

            # pick the value column
            val_col = None
            for cand in prefer_values:
                if cand in df.columns:
                    val_col = cand
                    break
            if not val_col:
                others = [c for c in df.columns if c.lower() != "date"]
                if len(others) == 1:
                    val_col = others[0]
                else:
                    raise ValueError(f"Could not identify value column. Got {list(df.columns)}")

            out = df[["date", val_col]].copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out[val_col] = pd.to_numeric(out[val_col], errors="coerce")
            out = out.dropna(subset=["date", val_col]).sort_values("date")
            return out.rename(columns={val_col: "value"})

        # ---------- parse payload ----------
        hist_raw = parse_block(js, "history", "history_columns")
        fcst_raw = parse_block(js, "forecast", "forecast_columns")

        hist_df = coerce_series(hist_raw, prefer_values=("sales", "value", "y"))
        hist_df = hist_df.rename(columns={"value": "History"}).set_index("date")

        if not fcst_raw.empty:
            fcst_df = coerce_series(fcst_raw, prefer_values=("sales_hat", "forecast", "yhat", "value"))
            fcst_df = fcst_df.rename(columns={"value": "Forecast"}).set_index("date")
        else:
            fcst_df = pd.DataFrame(columns=["Forecast"])
            fcst_df.index.name = "date"

        # quick sanity (counts + ranges)
        st.caption(
            f"hist rows={len(hist_df)}, fcst rows={len(fcst_df.dropna())} · "
            f"hist [{hist_df.index.min().date() if not hist_df.empty else 'N/A'} → "
            f"{hist_df.index.max().date() if not hist_df.empty else 'N/A'}]"
            + (f" · fcst [{fcst_df.index.min().date()} → {fcst_df.index.max().date()}]" if not fcst_df.empty else "")
        )

        # ---------- build plot data with bridge ----------
        hist_plot = hist_df.reset_index().rename(columns={"History": "value"})
        hist_plot["series"] = "History"

        if not fcst_df.empty and not hist_df.empty:
            last_hist_date = hist_df.index.max()
            last_hist_value = hist_df.loc[last_hist_date, "History"]

            fcst_plot = fcst_df.reset_index().rename(columns={"Forecast": "value"})
            fcst_plot["series"] = "Forecast"

            bridge = pd.DataFrame({"date": [last_hist_date], "value": [last_hist_value], "series": ["Forecast"]})
            fcst_plot = pd.concat([bridge, fcst_plot], ignore_index=True).drop_duplicates(subset=["date"], keep="first")

            plot_df = pd.concat([hist_plot, fcst_plot], ignore_index=True)
        elif not fcst_df.empty:
            fcst_plot = fcst_df.reset_index().rename(columns={"Forecast": "value"})
            fcst_plot["series"] = "Forecast"
            plot_df = pd.concat([hist_plot, fcst_plot], ignore_index=True)
        else:
            plot_df = hist_plot

        plot_df = plot_df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)

        # Optional: only show recent
        if show_recent and not hist_df.empty:
            cutoff_date = hist_df.index.max() - pd.Timedelta(days=90)
            plot_df = plot_df[plot_df["date"] >= cutoff_date]

        if plot_df.empty:
            st.info("No data to plot after parsing.")
            with st.expander("Payload (debug)"):
                st.json(js)
            st.stop()

        last_hist = hist_df.index.max() if not hist_df.empty else None

        # ---------- Altair chart ----------
        base = alt.Chart(plot_df).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Sales"),
            color=alt.Color(
                "series:N",
                title="Series",
                scale=alt.Scale(domain=["History", "Forecast"], range=["#1f77b4", "#ff7f0e"]),
            ),
            tooltip=[
                alt.Tooltip("date:T", format="%Y-%m-%d"),
                alt.Tooltip("series:N"),
                alt.Tooltip("value:Q", format=",.0f", title="Sales"),
            ],
        )

        hist_line = alt.Chart(plot_df[plot_df["series"] == "History"]).mark_line(size=2, color="#1f77b4").encode(
            x="date:T", y="value:Q",
            tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"),
                     alt.Tooltip("value:Q", format=",.0f", title="History")]
        )

        fcst_line = alt.Chart(plot_df[plot_df["series"] == "Forecast"]).mark_line(
            size=2, color="#ff7f0e", strokeDash=[6, 4]
        ).encode(
            x="date:T", y="value:Q",
            tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"),
                     alt.Tooltip("value:Q", format=",.0f", title="Forecast")]
        )

        chart = hist_line + fcst_line

        forecast_points = plot_df[plot_df["series"] == "Forecast"]
        if not forecast_points.empty:
            pts = alt.Chart(forecast_points).mark_circle(size=50, opacity=0.9).encode(
                x="date:T", y="value:Q", color=alt.value("#ff7f0e"),
                tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"),
                         alt.Tooltip("value:Q", format=",.0f", title="Forecast")]
            )
            chart = chart + pts

        if last_hist is not None:
            cutoff = alt.Chart(pd.DataFrame({"date": [last_hist]})).mark_rule(
                color="#666", strokeDash=[3, 3], size=1
            ).encode(x="date:T")
            chart = chart + cutoff

        st.altair_chart(chart.properties(height=400).interactive(), use_container_width=True)

        _show_win = js.get('model') in ('ma7_baseline', 'drift')
        st.caption(
            f"Model: **{js.get('model','n/a')}**"
            + (f" (window={int(window)}d)" if _show_win else "")
            + f". History through {last_hist.date() if last_hist is not None else 'N/A'}. "
              f"Forecast shows {len(fcst_df)} days ahead."
        )

        with st.expander("🔍 Debug Info"):
            st.write("**Last 3 history rows:**")
            st.dataframe(hist_df.tail(3))
            st.write("**First 3 forecast rows:**")
            st.dataframe(fcst_df.head(3))
            st.write("**Plot dataframe shape:**", plot_df.shape)
            st.write("**Forecast rows in plot:**", len(plot_df[plot_df["series"] == "Forecast"]))

    except Exception as e:
        st.error(f"Forecast error: {e}")
        import traceback
        with st.expander("Full error traceback"):
            st.code(traceback.format_exc())

# -------------------- Anomalies --------------------
with tabs[4]:
    st.header("Anomalies")
    st.write("Detected anomalies will be listed here.")

# -------------------- Ask AI --------------------
with tabs[5]:
    st.header("Ask AI")
    CLIENT_TIMEOUT = int(os.getenv("CLIENT_TIMEOUT", "55"))
    from pathlib import Path

    def load_sample_prompts() -> list[str]:
        defaults = [
            # Data-focused
            "Show avg satisfaction by region for the two most recent quarters.",
            "How did satisfaction change in the North region last quarter?",
            "What are the monthly sales trends for each product over the entire time period?",
            "Compare year-over-year sales performance by quarter. Which periods showed the strongest growth or decline?",
            "What is the correlation between transaction value and customer satisfaction?",
            "List the top 3 products by total sales for customers under 30 years old.",
            "Identify regions where sales are increasing but customer satisfaction is decreasing.",
            "Compare year-over-year sales performance by quarter. Which periods showed the strongest growth or decline?",
            "Analyze customer satisfaction scores across different age groups. Are there specific age segments that are consistently more or less satisfied?",
            "What month showed the highest overall sales growth?",
            "Which regions have growing sales but declining satisfaction?",
            "Which regions consistently outperform others in sales, and what factors might contribute to this success?",
            "Are there any correlations between gender and average satisfaction?",
            "What positive trends are evident in each of the regions?",
            # PDF-focused
            "Summarize the key ideas from the Walmart PDF.",
            "How can AI be a core component of value creation in a business model?",
            "From the PDF: what are the main drivers of sales growth and how are they measured?",
            "From the PDF: list three risks the analysis highlights and their mitigation approaches.",
            "What are some of the domains that are accepting of time series analysis and predictions?"
        ]
        p = Path("prompts.txt")
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8")
                # one prompt per non-empty line
                lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                # de-dup, keep order
                seen = set()
                merged = []
                for x in lines + defaults:
                    if x not in seen:
                        merged.append(x); seen.add(x)
                return merged
            except Exception:
                return defaults
        return defaults

    samples = load_sample_prompts()

    # Ensure session key exists BEFORE rendering the widget
    if "ask_query" not in st.session_state:
        st.session_state["ask_query"] = ""

    # Row with a sample select and insert button
    scols = st.columns([3, 1])
    with scols[0]:
        sample_choice = st.selectbox("Try a sample question", ["(choose a prompt)"] + samples, index=0)
    with scols[1]:
        if st.button("Insert", use_container_width=True, type="secondary", disabled=(sample_choice == "(choose a prompt)")):
            st.session_state["ask_query"] = sample_choice

    # The actual input (wired to session_state so Insert populates it)
    q = st.text_input("Ask a question about the data or the docs:", key="ask_query", placeholder="Type or insert a sample…")
    # q = st.text_input("Ask a question about the data or the docs:", value=st.session_state.get("ask_query", ""), key="ask_query")
    topk = st.slider("Citations to retrieve", 1, 10, 3, 1, key="askai_k")
    ask_clicked = st.button("Ask", key="askai_btn")
    if ask_clicked and q.strip():
        # Do the network call with a spinner only; render AFTER this block
        with st.spinner("Retrieving context & generating answer…"):
            try:
                r = requests.post(
                    f"{API_URL}/rag/query",
                    json={"query": q, "k": topk},
                    timeout=CLIENT_TIMEOUT,
                )
                body = r.json() if r.status_code == 200 else {}
                http_err = None if r.status_code == 200 else r.text
            except requests.Timeout:
                body, http_err = None, "timeout"
            except Exception as e:
                body, http_err = None, str(e)

        # ---------- RENDER RESULTS (no extra click required) ----------
        if http_err == "timeout":
            st.error("The request timed out. Try a shorter question or re-run.")
        elif http_err and not body:
            st.error(f"Error: {http_err}")
        else:
            body = body or {}

            # --- Status badge + route reason ---
            def _badge(kind: str):
                colors = {"duckdb": "green", "llm-sql": "blue", "docs": "violet", "fallback": "gray"}
                c = colors.get((kind or "fallback").lower(), "gray")
                st.markdown(
                    f"<span style='background:{c};color:white;padding:4px 8px;border-radius:8px;'>"
                    f"{(kind or 'fallback').upper()}</span>",
                    unsafe_allow_html=True,
                )

            cols = st.columns([1, 3])
            with cols[0]:
                _badge(body.get("source_used"))
            with cols[1]:
                if body.get("route_reason"):
                    st.caption(body["route_reason"])

            # --- Main answer ---
            st.markdown(body.get("answer", ""))

            # --- Optional SQL used (no expander to avoid nesting issues) ---
            if body.get("sql"):
                st.caption("SQL used")
                st.code(body["sql"], language="sql")

            # --- Optional table preview (supports {"headers","rows"} OR list[dict]) ---
            table = body.get("table")
            table_preview = body.get("table_preview")
            try:
                if isinstance(table, dict) and table.get("rows"):
                    st.dataframe(pd.DataFrame(table["rows"], columns=table.get("headers")))
                elif isinstance(table_preview, list) and table_preview:
                    st.dataframe(pd.DataFrame(table_preview))
            except Exception:
                pass

            # --- Sources / Citations (INSIDE the else:, so `body` exists) ---
            cites = body.get("citations", []) or []
            data_sources = body.get("data_sources", []) or []
            source_used = (body.get("source_used") or "").lower()

            if source_used == "docs" and cites:
                st.divider()
                st.caption("Citations")
                for c in cites:
                    if isinstance(c, dict):
                        idx = c.get("index", "")
                        src = c.get("source", "")
                        pg  = c.get("page")
                        ch  = c.get("chunk")
                        page_str  = f" p.{pg}" if pg not in (None, "", "None") else ""
                        chunk_str = f" c.{ch}" if ch not in (None, "", "None") else ""
                        st.write(f"[{idx}] {src}{page_str}{chunk_str}")
                    else:
                        st.write(str(c))
            elif source_used == "duckdb":
                if isinstance(data_sources, list) and data_sources:
                    st.caption("Data sources")
                    for s in data_sources:
                        st.write(f"• {s}")

# -------------------- System --------------------
with tabs[6]:
    st.header("System")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(f"Health check failed: {e}")
