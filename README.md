# Retail Insights Assistant — GenAI + Scalable Data System

A senior-level multi-agent GenAI application that helps business users analyze retail sales data through natural-language questions and automated summaries. The system uses Gemini for language understanding and insight generation, LangGraph for agent orchestration, DuckDB for structured analytics, and Streamlit for the user interface.

## Assignment Coverage

| Requirement | Implementation |
|---|---|
| Conversational Q&A mode | Streamlit chatbot accepts ad-hoc business questions and returns business-friendly answers. |
| Summarization mode | Summary mode generates an executive-level overview of sales performance. |
| Multi-agent workflow | LangGraph workflow with Query Resolution, Data Extraction, Validation, and Response Generation agents. |
| Data layer | DuckDB loads and queries sample CSV files from `Sales Dataset/`. |
| LLM integration | Gemini API via `google-genai`, with model fallback and local DuckDB fallback for quota/key issues. |
| Prompt engineering | Agent-specific prompts for SQL generation, validation, and final business response formatting. |
| Conversation context | UI passes prior Q&A history into the orchestrator for context-aware follow-up handling. |
| 100GB+ scale design | Covered in `docs/DATA_FLOW_AND_100GB_SCALE.md` and `docs/Architecture_Presentation.pptx`. |
| Demo evidence | Screenshots included in `screenshots/`. |

## Project Structure

```text
retail_insight_architecture/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── .gitattributes
├── README.md
│
├── docs/
│   ├── Architecture_Presentation.pdf
│   ├── TECHNICAL_NOTES.md
│   ├── DATA_FLOW_AND_100GB_SCALE.md
│   ├── DEMO_EVIDENCE_GUIDE.md
│   ├── SUBMISSION_CHECKLIST.md
│   └── PROJECT_STRUCTURE.md
│
├── screenshots/
│   ├── 01.Streamlit_UI.png
│   ├── 02.streamlit_ui_after_gemini_api_activation.png
│   ├── 03.Which category has the highest sales.png
│   ├── 04.What is the total revenue by state.png
│   ├── 05.Show me top performing products..png
│   ├── 06.Screenshot 2026-Which region saw the most orders05-12 213959.png
│   ├── 07.Which region saw the most orders.png
│   └── 08.Generate an overall sales summary.png
│
├── Sales Dataset/
│   ├── Amazon Sale Report.csv
│   ├── Sale Report.csv
│   ├── International sale Report.csv
│   └── supporting CSV files
│
└── retail_insights/
    ├── __init__.py
    ├── agents.py
    ├── config.py
    ├── data_processor.py
    ├── llm_provider.py
    ├── orchestrator.py
    │
    └── ui/
        ├── __init__.py
        └── streamlit_app.py
```

> `venv/`, `.env`, `__pycache__/`, and generated DuckDB files are intentionally not included in the submission zip. Create `venv/` locally during setup.

## Setup and Run

### 1. Open the project folder

```bash
cd retail_insight_architecture
```

### 2. Create and activate virtual environment

Windows CMD:

```bash
python -m venv venv
venv\Scripts\activate
```

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env`

Windows CMD:

```bash
copy .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Add your Gemini key in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash
```

You can also paste the Gemini key directly in the Streamlit UI. The UI key is session-only and overrides the backend key for that session.

### 5. Run the application

```bash
streamlit run app.py
```

If port `8501` is busy:

```bash
streamlit run app.py --server.port 8502
```

Open:

```text
http://localhost:8501
```

or, if using custom port:

```text
http://localhost:8502
```

## Example Questions

- Which category has the highest sales?
- What is the total revenue by state?
- Show me top performing products.
- Which region saw the most orders?
- What are the sales trends?
- Generate an overall sales summary.

## Architecture Summary

The application follows a layered architecture:

1. **Streamlit UI** captures user intent, API key, query mode, and conversation history.
2. **Orchestrator** coordinates the LangGraph workflow.
3. **Query Resolution Agent** converts natural language into a structured analytical plan and SQL where applicable.
4. **Data Extraction Agent** executes DuckDB queries or summary aggregations.
5. **Validation Agent** checks whether results answer the question and flags errors or low-confidence outputs.
6. **Response Generation Agent** converts validated data into executive-ready business language.
7. **DuckDB Data Layer** loads CSV files and serves fast analytical queries.

## Core Technologies

- Python
- Streamlit
- Gemini API (`google-genai`)
- LangGraph / LangChain Core
- DuckDB
- Pandas
- Plotly
- python-dotenv

## Scalability Design

The current application runs locally on sample CSV files. For 100GB+ production scale, the proposed design uses:

- Object storage such as S3, ADLS, or GCS for raw and curated data.
- Delta Lake / Apache Iceberg / Parquet tables for partitioned analytical storage.
- Spark, Databricks, BigQuery, or Snowflake for distributed preprocessing and SQL execution.
- Metadata-aware query planning to reduce scanned data.
- Optional vector indexing for report/document retrieval.
- Prompt caching, result caching, and query validation to control LLM cost and latency.

Full details are available in `docs/DATA_FLOW_AND_100GB_SCALE.md`.

## Assumptions

- The sample dataset is available under `Sales Dataset/`.
- Gemini API key is provided through `.env` or the Streamlit UI.
- The local version is optimized for assignment/demo usage, not direct production deployment.
- DuckDB is used for local structured analytics; distributed execution is proposed for 100GB+ scale.

## Limitations

- The demo dataset may not contain all columns needed for every advanced analytical question, such as complete YoY comparisons across multiple years.
- LLM output quality depends on API availability, model behavior, and prompt adherence.
- The UI does not currently support file upload at runtime; it reads CSV files from the project dataset folder.
- Authentication, role-based access control, and audit logging are outside the local demo scope.

## Possible Improvements

- Add runtime CSV upload and schema profiling.
- Add semantic layer / metric definitions for business-approved KPIs.
- Add persistent conversation memory.
- Add automated evaluation using golden SQL/query-answer pairs.
- Deploy with containerization and cloud data warehouse integration.
- Add observability for latency, cost, token usage, and validation failure rate.

## Submission Contents

- Working source code
- Sample sales CSV data
- README and technical notes
- 100GB scale architecture document
- Mandatory architecture presentation
- Demo screenshots
