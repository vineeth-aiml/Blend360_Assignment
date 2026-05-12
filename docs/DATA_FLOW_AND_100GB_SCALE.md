# Data Flow and 100GB+ Scale Architecture

## Current Local Data Flow

```text
User Question
   ↓
Streamlit UI
   ↓
LangGraph Orchestrator
   ↓
Query Resolution Agent
   ↓
DuckDB SQL / Summary Aggregation
   ↓
Validation Agent
   ↓
Response Generation Agent
   ↓
Business Answer + SQL + Table/Chart
```

## Current Local Storage

The demo version reads CSV files from `Sales Dataset/` and loads them into DuckDB tables. This is ideal for a local assignment because it is simple, fast, and easy to reproduce.

## Production-Scale Target

When historical sales data grows beyond 100GB, direct local CSV scanning is not sufficient. The scalable architecture should separate ingestion, storage, query planning, retrieval, LLM orchestration, and monitoring.

## A. Data Engineering and Preprocessing

### Batch Ingestion

- Land raw sales files in object storage such as AWS S3, Azure Data Lake, or Google Cloud Storage.
- Use Spark, Databricks, BigQuery Dataflow, or Dask for large CSV ingestion.
- Validate schema, normalize column names, standardize region/state/product fields, and handle nulls.
- Write curated outputs as partitioned Parquet/Delta/Iceberg tables.

### Streaming or Incremental Ingestion

- Use Kafka/Kinesis/Pub/Sub for transactional sales events.
- Write bronze/raw data first, then silver/cleaned and gold/business aggregates.
- Maintain incremental aggregates by date, region, product, category, and channel.

## B. Storage and Indexing

### Recommended Storage Layers

| Layer | Purpose | Example Technology |
|---|---|---|
| Bronze | Raw immutable data | S3 / ADLS / GCS |
| Silver | Cleaned normalized tables | Delta Lake / Iceberg / Parquet |
| Gold | Business aggregates and KPI tables | BigQuery / Snowflake / Databricks SQL |
| Semantic Layer | Metric definitions and business glossary | dbt metrics / Cube / custom metadata service |
| Vector Index | Search over reports, definitions, docs | FAISS / Pinecone / ChromaDB |

### Partitioning Strategy

Partition large fact tables by:

- `year`, `month`, `date`
- `region` or `market`
- `sales_channel`

Cluster or sort by:

- `product_id`
- `category`
- `customer_id`
- `order_id`

## C. Retrieval and Query Efficiency

The assistant should not send large raw datasets to the LLM. Instead:

1. Use the LLM only to interpret intent and generate a constrained analytical plan.
2. Use SQL engines to retrieve only relevant rows/aggregates.
3. Apply metadata filtering before SQL generation.
4. Return small result sets to the LLM for final explanation.
5. Cache repeated query plans and SQL results.

### Retrieval Pattern

```text
Natural Language Query
   ↓
Intent + Entity Extraction
   ↓
Metadata/Semantic Filter
   ↓
SQL Generation Against Approved Schema
   ↓
Warehouse/Lakehouse Execution
   ↓
Small Aggregated Result
   ↓
Validated Business Response
```

## D. RAG and Vector Indexing

For structured sales facts, SQL is the primary retrieval mechanism. Vector search is useful for:

- Business glossary lookup
- KPI definitions
- Historical reports
- Data dictionary retrieval
- Policy or domain context

A hybrid RAG design can retrieve business definitions from a vector store and structured metrics from the warehouse.

## E. Model Orchestration at Scale

### Cost and Latency Controls

- Use small/fast models for query classification and schema routing.
- Use stronger models only for complex reasoning or final narrative generation.
- Cache prompt outputs and SQL result sets.
- Add SQL guardrails to prevent full table scans unless explicitly approved.
- Limit result rows before passing data to the LLM.

### LangGraph Flow at Scale

```text
Router Agent
   ↓
Schema/Metric Retrieval Agent
   ↓
SQL Generation Agent
   ↓
Query Execution Agent
   ↓
Validation Agent
   ↓
Response Agent
   ↓
Monitoring + Feedback Store
```

## F. Monitoring and Evaluation

Track:

- SQL execution latency
- LLM latency
- Token usage and estimated cost
- Cache hit rate
- Query failure rate
- Validation confidence
- User feedback score
- Hallucination or unsupported-answer rate

## G. Low-Confidence Handling

If validation confidence is low:

- Ask a clarifying question.
- Show the generated SQL and extracted evidence.
- Fall back to deterministic aggregates.
- Refuse unsupported claims when data is insufficient.

## H. Production Reference Architecture

```text
Sales Sources / CSV / APIs / Events
   ↓
Ingestion Layer: Spark / Databricks / Dataflow
   ↓
Lakehouse: S3/ADLS/GCS + Delta/Iceberg/Parquet
   ↓
Warehouse / SQL Engine: Snowflake / BigQuery / Databricks SQL
   ↓
Metadata + Semantic Layer: schemas, KPIs, glossary
   ↓
LangGraph Agent Orchestration
   ↓
Gemini/OpenAI LLM + Prompt Templates + Cache
   ↓
Streamlit / Web App / BI Interface
   ↓
Monitoring, Evaluation, Audit Logs
```
