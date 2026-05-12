# Technical Notes

## Objective

Build a GenAI-powered Retail Insights Assistant that can answer ad-hoc analytical questions and generate summaries from structured retail sales data.

## Implementation Overview

The solution is implemented as a Python Streamlit application with a LangGraph-based multi-agent backend.

### Main Components

| Component | Responsibility |
|---|---|
| `app.py` | Thin entrypoint that launches the Streamlit UI. |
| `retail_insights/ui/streamlit_app.py` | User interface, API key handling, chat input, summary/Q&A mode, result visualization. |
| `retail_insights/orchestrator.py` | Coordinates the agent workflow using LangGraph. |
| `retail_insights/agents.py` | Implements the four core agents and local fallback responses. |
| `retail_insights/data_processor.py` | Loads CSV files into DuckDB, profiles tables, executes SQL, and generates summary statistics. |
| `retail_insights/llm_provider.py` | Gemini adapter with retry, fallback model, and in-memory prompt cache. |
| `retail_insights/config.py` | Loads environment variables from `.env`. |

## Multi-Agent Design

### 1. Query Resolution Agent

Converts user intent into a structured analytical plan. For Q&A queries, it generates SQL where the answer can be derived from tabular data.

### 2. Data Extraction Agent

Runs SQL through DuckDB or generates curated summary statistics for summary mode.

### 3. Validation Agent

Checks if extracted data is relevant, complete, and sufficient for the user query.

### 4. Response Generation Agent

Converts the validated result into a concise business-facing answer. If Gemini quota/key issues occur, the app falls back to a local DuckDB-driven answer instead of failing completely.

## Data Layer

DuckDB is used because it can query CSV files efficiently, supports SQL analytics, and is lightweight enough for a local assignment demo. The local data layer loads:

- `Amazon Sale Report.csv` as `amazon_sales`
- `Sale Report.csv` as `inventory`
- `International sale Report.csv` as `international_sales`

## Prompt Engineering

The prompt layer is agent-specific:

- Query resolution prompts include schema context, sample rows, table names, and business rules.
- Validation prompts focus on answerability and result quality.
- Response prompts ask for concise, executive-style business language.

## Conversation Context

The UI keeps recent Q&A turns in session state and sends them to the orchestrator so follow-up questions can reuse prior context.

## Error Handling and Fallbacks

The Gemini adapter detects quota, billing, invalid key, and server-side errors. It avoids repeated quota retries, tries the fallback model when configured, and allows the agent layer to generate a local DuckDB response when model output is unavailable.

## Assumptions

- CSV files are already present in `Sales Dataset/`.
- The user provides a valid Gemini API key.
- Local execution is acceptable for the demo dataset.
- For 100GB+ scale, the system would move from local DuckDB/CSV to lakehouse or warehouse-backed analytics.

## Limitations

- No production authentication or authorization layer.
- No persistent user/session memory outside Streamlit runtime.
- No runtime file-upload pipeline in the current UI.
- Advanced queries requiring unavailable columns or time periods may need schema expansion.

## Future Enhancements

- Add file upload and automatic schema detection.
- Add semantic metric layer for approved business KPIs.
- Add golden-query test suite for evaluation.
- Add observability for latency, token usage, cost, and validation failure rate.
- Add cloud deployment using Docker and managed data warehouse integration.
