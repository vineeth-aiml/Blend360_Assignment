# Project Structure

```text
retail_insight_architecture/
│
├── app.py                              # Streamlit launcher
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── .gitignore                          # Excludes local/generated files
├── README.md                           # Setup, execution, architecture notes
│
├── docs/
│   ├── Architecture_Presentation.pptx  # Mandatory architecture deck
│   ├── TECHNICAL_NOTES.md              # Engineering notes and assumptions
│   ├── DATA_FLOW_AND_100GB_SCALE.md    # 100GB+ scalable architecture design
│   ├── DEMO_EVIDENCE_GUIDE.md          # Screenshots and testing guide
│   ├── SUBMISSION_CHECKLIST.md         # Deliverables checklist
│   └── PROJECT_STRUCTURE.md            # This file
│
├── screenshots/                        # Demo evidence captured from Streamlit UI
│
├── Sales Dataset/                      # Sample CSV files used by DuckDB
│
└── retail_insights/
    ├── __init__.py
    ├── agents.py                       # Query, extraction, validation, response agents
    ├── config.py                       # Environment and model configuration
    ├── data_processor.py               # DuckDB loading, schema, summaries, SQL execution
    ├── llm_provider.py                 # Gemini adapter, retry/fallback/cache logic
    ├── orchestrator.py                 # LangGraph multi-agent orchestration
    └── ui/
        ├── __init__.py
        └── streamlit_app.py            # Streamlit UI implementation
```

## Not Included in Submission Zip

```text
venv/                 # recreate locally with python -m venv venv
.env                  # local secret file; create from .env.example
__pycache__/          # Python runtime cache
*.duckdb              # generated local DuckDB files
*.log                 # runtime logs
```
