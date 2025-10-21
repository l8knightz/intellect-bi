# 🧠 Intellect BI
**An AI-powered Business Intelligence assistant for analytics, forecasting, and document insight.**

Intellect BI merges **data analytics**, **retrieval-augmented generation (RAG)**, and **forecast modeling** into one containerized workspace.  
It enables natural-language querying of structured (CSV) and unstructured (PDF) data sources, returning visual summaries, metrics, and forecasts—all powered by local LLM inference.

---

## 🚀 Features
| Capability | Description |
|-------------|--------------|
| **Ask AI** | Query your sales data or business PDFs in plain English. The router decides whether to use DuckDB (for data) or RAG (for docs). |
| **Data Awareness** | Automatically detects schema terms (sales, region, quarter, satisfaction) to generate safe SQL queries against the CSV dataset. |
| **Document Insight** | Retrieves and summarizes information from uploaded PDFs using ChromaDB embeddings. |
| **Forecast Tab** | Provides 7-day / 30-day forecasts with MA7 and Seasonal models using Altair charts. |
| **Drill-Down & Anomalies** | Explore regions, quarters, and satisfaction trends with visual breakdowns and anomaly highlights. |
| **Multi-tab Streamlit UI** | Interactive interface with overview, drill-down, insights, anomalies, and forecast tabs. |
| **Offline Friendly** | Runs fully local with Ollama or other LLM endpoints—no external API required. |

---

## 🧩 Architecture Overview
```
+--------------------------+
| Streamlit UI |
| (ui/app.py, logo, tabs) |
+------------+-------------+
|
v
+------------+-------------+
| FastAPI backend (api) |
| - Router (route.py) |
| - RAG / DuckDB engine |
| - Forecast models |
+------------+-------------+
|
+-------+-------+
| ChromaDB |
| (vector store)|
+---------------+
|
+-------+-------+
| Ollama (LLM) |
| local model |
+---------------+
```
---

## 🧠 Core Components

| Folder | Purpose |
|---------|----------|
| `/api` | FastAPI backend — handles `/rag/query`, `/analytics/forecast`, and routing between data/doc modes. |
| `/ui` | Streamlit application with multi-tab layout and embedded logo. |
| `/data` | Example structured dataset (`sales_data.csv`). |
| `/docs` | Uploaded PDFs for unstructured queries. |
| `/models` | Forecast and embedding model utilities. |
| `/vectorstore` | Persistent ChromaDB database. |

## Repo Structure
```
intellect-bi/
├── api/
│   ├── main.py
│   ├── router.py
│   └── ...
├── ui/
│   ├── app.py
│   ├── assets/logo.png
│   └── ...
├── data/sales_data.csv
├── docs/ (PDF sources)
├── vectorstore/
├── docker-compose.yml
└── README.md
```

---

## 🐳 Quick Start (Local / Docker)

### 1. Prerequisites
- Docker Desktop or Docker Engine + Compose v2  
- (Optional) [Ollama](https://ollama.ai) installed locally with `phi3:mini` or `mistral` model pulled  
- Python ≥ 3.10 (if running manually)

### 2. Clone the Repository
```bash
git clone https://github.com/<your-user>/intellect-bi.git
cd intellect-bi
```

### 3. Start the Stack
```
docker compose up -d
```

Should launch:
 - backend → FastAPI on port 8000
 - frontend → Streamlit UI on port 8501
 - ollama → Local LLM endpoint on port 11434
 - chromadb → Persistent vector store on port 8001

View the app at http://localhost:8501

## 🧾 Services in docker-compose.yml
| Service    | Port  | Description                                                  |
| ---------- | ----- | ------------------------------------------------------------ |
| `ollama`   | 11434 | Hosts the local language model used for query generation.    |
| `api`      | 8000  | FastAPI service handling data queries, RAG, and forecasting. |
| `ui`       | 8501  | Streamlit front-end with multi-tab dashboard and logo.       |
| `chromadb` | 8001  | Vector store for embedding and retrieval of PDF data.        |

## Environment Variables
| Variable           | Description                         | Example                |
| ------------------ | ----------------------------------- | ---------------------- |
| `OLLAMA_URL`       | Endpoint for local LLM inference    | `http://ollama:11434`  |
| `CHROMA_PATH`      | Persistent path for ChromaDB        | `/vectorstore`         |
| `DATA_PATH`        | Location of the main CSV dataset    | `/data/sales_data.csv` |
| `DOC_PATH`         | Directory containing PDFs           | `/docs`                |
| `ASK_AI_SQL_LIMIT` | Row limit for generated SQL queries | `200`                  |

## Example API Calls
Query sales data
```
curl -s -X POST http://localhost:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare year-over-year sales performance by quarter."}' | jq
```
Generate forecast
```
curl -s "http://localhost:8000/analytics/forecast?horizon=30" | jq '.model, .forecast[0]'
```

## 🧮 Data Sources
 - sales_data.csv — Synthetic dataset for demonstration.
 - PDFs under /docs — Example market reports and summaries.
 - ChromaDB vector store stores embeddings for these PDFs.

## Routing Logic Summary
| Query Type                                            | Example            | Route  | Source           |
| ----------------------------------------------------- | ------------------ | ------ | ---------------- |
| “Average satisfaction by region”                      | metrics & tables   | DuckDB | `sales_data.csv` |
| “Summarize key ideas from Walmart PDF”                | descriptive        | RAG    | Uploaded PDFs    |
| “Compare year-over-year sales performance by quarter” | numeric, quarterly | DuckDB | Template SQL     |
| “Explain challenges in supply chain management”       | contextual         | RAG    | PDFs             |

## 🧰 Development Workflow
Backend is FastAPI
```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Frontend with Streamlit
```
streamlit run ui/app.py
```
You can use these to reload both services

## 🧼 Data Safety & Query Guardrails
 - Only SELECT statements are allowed in SQL generation.
 - Automatic sanitation prevents mutation or DDL statements.
 - YOY-by-quarter and similar metrics use deterministic templates for reliability.
 - No document fallback occurs when data routes are intended.

## 📊 Example Output
Ask AI → “Compare year-over-year sales performance by quarter.”
```
Strongest growth: 2023 Q2 (+12.8%)
Largest decline: 2024 Q1 (-6.4%)
```

## How can you expand/enhance it? (I may attempt these eventually)
 - Fine-tune the prompts to improve data/doc pathing
 - Add LSTM or Prophet forecasting
 - Include export options for visualization (ie; generate pngs or pdfs?)
 - OAuth with user profiles, save insights, upload data, etc.

## 🧱 Tech Stack
| Category         | Tools                   |
| ---------------- | ----------------------- |
| Backend          | FastAPI, DuckDB, Pandas |
| Frontend         | Streamlit, Altair       |
| AI / RAG         | Ollama, ChromaDB        |
| Forecasting      | MA7, Seasonal7 models   |
| Containerization | Docker Compose          |
| Language         | Python 3.10+            |

## 🧑‍💻 Authors & Credits
Travis B.

Senior Cloud / DevOps Engineer • Application Modernization • Generative AI Practitioner

Project developed as part of the Generative AI Specialization Capstone (Purdue University).