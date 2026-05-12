# Demo Evidence Guide

The `screenshots/` folder contains Streamlit UI evidence captured from the running application.

## Included Screenshots

| File | Evidence |
|---|---|
| `01_header_api_key.png` | Main UI header and API key activation section. |
| `02_view_questions.png` | Header-level View Questions menu with curated example prompts. |
| `03_qa_response_sql.png` | Conversational Q&A answer with generated SQL and extracted evidence. |
| `04_category_chart.png` | Chart output for highest-sales category. |
| `05_state_revenue_chart.png` | Regional/state revenue visualization. |
| `06_extracted_table.png` | Extracted table output for query evidence. |
| `07_api_key_empty_state.png` | Empty-state API key handling. |

## Suggested Demo Flow

1. Start the app with `streamlit run app.py`.
2. Add Gemini API key using the UI or `.env` file.
3. Open **View Questions** and select a curated prompt.
4. Run a Q&A query such as `Which category has the highest sales?`.
5. Open **View Details** to show generated SQL, extracted table, and chart.
6. Switch to Summary mode and generate an overall business summary.

## Example Q&A Evidence

Example query:

```text
Which category has the highest sales?
```

Expected answer pattern:

```text
The Set category has the highest sales, with total revenue around 35.7M.
```

Example query:

```text
What is the total revenue by state?
```

Expected answer pattern:

```text
Maharashtra leads total revenue, followed by Karnataka and Telangana.
```
