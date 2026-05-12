"""
Multi-Agent System using LangGraph
Implements: Query Resolution Agent, Data Extraction Agent, Validation Agent, Response Generation Agent
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from typing import Protocol
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
import logging
from retail_insights.data_processor import DataProcessor
import json


class LLMProtocol(Protocol):
    def invoke(self, messages, **kwargs):
        ...


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_quota_or_key_error(error_text: str) -> bool:
    """Detect Gemini quota/key failures so the app can fall back gracefully."""
    text = (error_text or "").upper()
    return any(
        marker in text
        for marker in (
            "429",
            "RESOURCE_EXHAUSTED",
            "QUOTA",
            "API_KEY_INVALID",
            "API KEY EXPIRED",
            "BILLING",
        )
    )


def _rule_based_query_resolution(user_query: str, tables: list[str]) -> dict:
    """Cheap fallback for common demo questions when Gemini quota is hit."""
    q = (user_query or "").lower()

    if any(word in q for word in ["trend", "trends", "summary", "summarize", "overall", "performance"]):
        return {
            "query_type": "summarization",
            "structured_query": user_query,
            "sql_query": None,
            "required_tables": ["amazon_sales"],
        }

    if "state" in q or "region" in q or "ship" in q:
        if "order" in q and not any(word in q for word in ["revenue", "sales", "amount"]):
            sql = """SELECT "ship-state" AS state, COUNT(*) AS order_count
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0 AND "ship-state" IS NOT NULL
GROUP BY "ship-state"
ORDER BY order_count DESC
LIMIT 20"""
        else:
            sql = """SELECT "ship-state" AS state, SUM(Amount) AS total_revenue, COUNT(*) AS order_count, AVG(Amount) AS avg_order_value
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0 AND "ship-state" IS NOT NULL
GROUP BY "ship-state"
ORDER BY total_revenue DESC
LIMIT 20"""
        return {
            "query_type": "qa",
            "structured_query": "Regional/state level sales performance",
            "sql_query": sql,
            "required_tables": ["amazon_sales"],
        }

    if "category" in q or "categories" in q or "product" in q:
        order_col = "avg_order_value" if any(word in q for word in ["average", "avg", "aov"]) else "total_revenue"
        sql = f"""SELECT Category, COUNT(*) AS order_count, SUM(Amount) AS total_revenue, AVG(Amount) AS avg_order_value
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0 AND Category IS NOT NULL
GROUP BY Category
ORDER BY {order_col} DESC
LIMIT 10"""
        return {
            "query_type": "qa",
            "structured_query": "Category level sales performance",
            "sql_query": sql,
            "required_tables": ["amazon_sales"],
        }

    return {
        "query_type": "qa",
        "structured_query": user_query,
        "sql_query": None,
        "required_tables": tables or ["amazon_sales"],
    }


def _format_number(value) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"
    except Exception:
        return str(value)


def _local_response_from_data(state: dict, error_text: str = "") -> str:
    """Create a useful local response when Gemini is unavailable/quota-limited."""
    data = state.get("extracted_data") or {}
    query = state.get("user_query", "your query")

    prefix = ""
    if _is_quota_or_key_error(error_text):
        prefix = "Gemini quota/rate limit was reached, so I used the local DuckDB result instead of asking the model again. "
    elif error_text:
        prefix = "The LLM response step failed, so I generated this answer from the extracted data. "

    if data.get("data_type") == "summary":
        summary_stats = data.get("summary_statistics", {})
        top_categories = data.get("top_categories", [])
        regional = data.get("regional_performance", [])

        parts = [prefix + "Sales summary from the available dataset:"]

        amazon_stats = summary_stats.get("amazon_sales", {}) if isinstance(summary_stats, dict) else {}
        total_revenue = (
            amazon_stats.get("total_revenue")
            or amazon_stats.get("revenue")
            or amazon_stats.get("total_sales")
        )
        total_orders = (
            amazon_stats.get("total_orders")
            or amazon_stats.get("orders")
            or amazon_stats.get("total_records")
        )

        if total_revenue:
            parts.append(f"Total revenue is around **{_format_number(total_revenue)}**.")
        if total_orders:
            parts.append(f"Total orders/records are around **{_format_number(total_orders)}**.")

        if top_categories:
            top = top_categories[0]
            category = top.get("Category") or top.get("category") or "top category"
            revenue = top.get("total_revenue") or top.get("total_sales") or top.get("total") or top.get("Amount")
            orders = top.get("order_count") or top.get("orders") or top.get("count")
            sentence = f"Top category is **{category}**"
            if revenue:
                sentence += f" with revenue around **{_format_number(revenue)}**"
            if orders:
                sentence += f" and **{_format_number(orders)}** orders"
            sentence += "."
            parts.append(sentence)

        if regional:
            top_region = regional[0]
            state_name = (
                top_region.get("ship-state")
                or top_region.get("state")
                or top_region.get("region")
                or "top region"
            )
            revenue = top_region.get("total_revenue") or top_region.get("total_sales") or top_region.get("Amount")
            sentence = f"Top region/state is **{state_name}**"
            if revenue:
                sentence += f" with revenue around **{_format_number(revenue)}**"
            sentence += "."
            parts.append(sentence)

        parts.append("For more detail, open **View Details** to inspect the extracted tables.")
        return " ".join(parts)

    rows = data.get("query_result") or data.get("fallback_data") or []
    if rows:
        first = rows[0]
        keys = list(first.keys())
        title_key = next(
            (k for k in keys if k.lower() in {"category", "state", "ship-state", "region", "sku"}),
            keys[0],
        )
        metric_keys = [k for k in keys if k != title_key]

        metrics = []
        for key in metric_keys[:3]:
            metrics.append(f"{key}: **{_format_number(first.get(key))}**")

        answer = prefix + f"For **{query}**, the top result is **{first.get(title_key)}**"
        if metrics:
            answer += " with " + ", ".join(metrics)
        answer += ". Open **View Details** to see the full extracted table."
        return answer

    return (
        prefix
        + "I could not generate a model response, but the workflow completed. "
        + "Please check the extracted data in **View Details** or try again after the Gemini quota resets."
    )


class AgentState(TypedDict):
    """State shared across all agents."""
    user_query: str
    query_type: str
    conversation_history: Optional[List[Dict[str, str]]]
    structured_query: Optional[str]
    sql_query: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
    errors: Annotated[List[str], operator.add]
    metadata: Dict[str, Any]


class QueryResolutionAgent:
    """
    Agent 1: Converts natural language queries into structured queries.
    """

    def __init__(self, llm: LLMProtocol, data_processor: DataProcessor):
        self.llm = llm
        self.data_processor = data_processor
        self.name = "QueryResolutionAgent"

    def _format_conversation_history(self, history: list) -> str:
        """Format recent conversation history for the LLM prompt."""
        if not history:
            return "No prior conversation."

        recent = history[-5:]
        formatted = []

        for i, exchange in enumerate(recent, 1):
            formatted.append(f"  Turn {i}:")
            formatted.append(f"    User: {exchange.get('query', '')}")

            resp = exchange.get("response", "")
            if len(resp) > 200:
                resp = resp[:200] + "..."

            formatted.append(f"    Assistant: {resp}")

        return "\n".join(formatted)

    def run(self, state: AgentState) -> AgentState:
        """
        Analyze user query and convert it to structured SQL/query metadata.
        """
        logger.info(f"[{self.name}] Processing user query...")

        try:
            tables = self.data_processor.get_available_tables()
            table_contexts = {
                table: self.data_processor.get_table_context(table)
                for table in tables
            }

            logger.info(f"[{self.name}] Retrieved table contexts for query resolution")
            logger.info(f"[{self.name}] Available tables: {tables}")

            system_prompt = f"""You are a Query Resolution Agent for a retail sales analytics system.

Your task is to analyze the user's natural language query and convert it into a valid SQL query.

DATABASE CONTEXT:
{json.dumps(table_contexts, indent=2, default=str)[:3500]}

KEY INSIGHTS:
- amazon_sales table has {table_contexts.get('amazon_sales', {}).get('statistics', {}).get('total_rows', 0):,} records
- Available Categories: {table_contexts.get('amazon_sales', {}).get('value_examples', {}).get('Category', [])}
- Date column is available in amazon_sales.

CRITICAL SQL Rules:
1. ALWAYS use proper GROUP BY clauses. Every non-aggregated column in SELECT must be in GROUP BY.
2. Use DuckDB SQL syntax.
3. Column names with spaces or special characters must be quoted with double quotes: "Order ID", "ship-state".
4. ALWAYS filter out cancelled orders: WHERE Status != 'Cancelled' AND Amount > 0.
5. Column names are case-sensitive. Use exact names from schema.
6. For summarization queries, set query_type to "summarization" and sql_query to null.

DATE HANDLING:
- Date column in amazon_sales is already DATE type in DuckDB.
- Extract year: YEAR(Date)
- Extract quarter: QUARTER(Date)
- Extract month: MONTH(Date)
- Do NOT use strptime(Date, ...).

YEAR-OVER-YEAR / GROWTH RATE PATTERN:
WITH sales_by_period AS (
    SELECT
        Category,
        YEAR(Date) AS year,
        SUM(Amount) AS revenue
    FROM amazon_sales
    WHERE Status != 'Cancelled' AND Amount > 0
    GROUP BY Category, year
),
yoy_calc AS (
    SELECT
        curr.Category,
        curr.year AS current_year,
        prev.year AS previous_year,
        curr.revenue AS current_revenue,
        prev.revenue AS previous_revenue,
        ((curr.revenue - prev.revenue) / prev.revenue * 100) AS yoy_growth_pct
    FROM sales_by_period curr
    LEFT JOIN sales_by_period prev
        ON curr.Category = prev.Category
        AND curr.year = prev.year + 1
)
SELECT Category, current_year, previous_year, yoy_growth_pct
FROM yoy_calc
WHERE previous_revenue IS NOT NULL
ORDER BY yoy_growth_pct DESC
LIMIT 1;

COMMON QUERY PATTERNS:
- Top categories:
SELECT Category, SUM(Amount) AS total_revenue
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0
GROUP BY Category
ORDER BY total_revenue DESC
LIMIT 10;

- Revenue by state:
SELECT "ship-state", SUM(Amount) AS total_revenue
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0
GROUP BY "ship-state"
ORDER BY total_revenue DESC;

- Orders by state:
SELECT "ship-state", COUNT(*) AS order_count
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0
GROUP BY "ship-state"
ORDER BY order_count DESC;

- Sales trends:
SELECT YEAR(Date) AS sales_year, MONTH(Date) AS sales_month, SUM(Amount) AS total_revenue
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0
GROUP BY sales_year, sales_month
ORDER BY sales_year, sales_month;

CONVERSATION CONTEXT:
{self._format_conversation_history(state.get('conversation_history', []))}

Return ONLY a JSON object. No markdown. No explanation.

JSON format:
{{
  "query_type": "summarization" or "qa",
  "structured_query": "description of what data is needed",
  "sql_query": "valid DuckDB SQL query or null for summarization",
  "required_tables": ["list", "of", "tables"]
}}

User Query: {state['user_query']}"""

            response = self.llm.invoke([
                SystemMessage(content=system_prompt)
            ])

            response_text = response.content

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError:
                parsed_response = {
                    "query_type": "qa",
                    "structured_query": state["user_query"],
                    "sql_query": None,
                    "required_tables": tables,
                }

            state["query_type"] = parsed_response.get("query_type", "qa")
            state["structured_query"] = parsed_response.get("structured_query", "")
            state["sql_query"] = parsed_response.get("sql_query", "")
            state["metadata"]["query_resolution"] = {
                "required_tables": parsed_response.get("required_tables", []),
                "success": True,
            }

            logger.info(f"[{self.name}] Query resolved - Type: {state['query_type']}")

        except Exception as e:
            error_text = str(e)
            logger.error(f"[{self.name}] Error: {error_text}")

            fallback = _rule_based_query_resolution(
                state["user_query"],
                tables if "tables" in locals() else [],
            )

            state["query_type"] = fallback.get("query_type", "qa")
            state["structured_query"] = fallback.get("structured_query", state["user_query"])
            state["sql_query"] = fallback.get("sql_query")
            state["metadata"]["query_resolution"] = {
                "success": False,
                "fallback_used": True,
                "error": error_text,
                "required_tables": fallback.get("required_tables", []),
            }

            if not _is_quota_or_key_error(error_text):
                state["errors"].append(f"Query Resolution Error: {error_text}")

        return state


class DataExtractionAgent:
    """
    Agent 2: Executes queries and extracts relevant data.
    """

    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.name = "DataExtractionAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Execute SQL query and extract data.
        """
        logger.info(f"[{self.name}] Extracting data...")

        try:
            extracted_data = {}

            if state["query_type"] == "summarization":
                summary_stats = self.data_processor.get_summary_statistics()
                top_categories = self.data_processor.get_top_categories(10)
                regional_performance = self.data_processor.get_regional_performance()

                extracted_data = {
                    "summary_statistics": summary_stats,
                    "top_categories": top_categories.to_dict("records") if not top_categories.empty else [],
                    "regional_performance": regional_performance.to_dict("records")[:10] if not regional_performance.empty else [],
                    "data_type": "summary",
                    "row_count": (
                        len(top_categories) if not top_categories.empty else 0
                    ),
                }

            elif state["sql_query"]:
                try:
                    result_df = self.data_processor.execute_query(state["sql_query"])
                    row_count = len(result_df)
                    dataset_metadata = self.data_processor.get_dataset_metadata()

                    extracted_data = {
                        "query_result": result_df.to_dict("records"),
                        "row_count": row_count,
                        "columns": list(result_df.columns),
                        "data_type": "query_result",
                        "dataset_metadata": dataset_metadata,
                    }

                    if row_count == 0:
                        logger.warning(f"[{self.name}] Query returned 0 rows - likely data limitation")
                        extracted_data["empty_result"] = True
                        extracted_data["user_query"] = state["user_query"]

                        fallback_df = self.data_processor.get_top_categories(10)
                        extracted_data["fallback_data"] = fallback_df.to_dict("records")
                        extracted_data["fallback_type"] = "category_stats"

                except Exception as sql_error:
                    logger.warning(f"[{self.name}] SQL query failed, using fallback: {str(sql_error)}")

                    query_lower = state["user_query"].lower()

                    if "category" in query_lower or "product" in query_lower:
                        fallback_df = self.data_processor.get_top_categories(10)
                        extracted_data = {
                            "query_result": fallback_df.to_dict("records"),
                            "row_count": len(fallback_df),
                            "columns": list(fallback_df.columns),
                            "data_type": "query_result",
                            "fallback": True,
                            "original_error": str(sql_error),
                        }

                    elif "region" in query_lower or "state" in query_lower:
                        fallback_df = self.data_processor.get_regional_performance()
                        extracted_data = {
                            "query_result": fallback_df.head(10).to_dict("records"),
                            "row_count": min(10, len(fallback_df)),
                            "columns": list(fallback_df.columns),
                            "data_type": "query_result",
                            "fallback": True,
                            "original_error": str(sql_error),
                        }

                    else:
                        summary_stats = self.data_processor.get_summary_statistics()
                        extracted_data = {
                            "query_result": [summary_stats.get("amazon_sales", {})],
                            "row_count": 1,
                            "columns": list(summary_stats.get("amazon_sales", {}).keys()),
                            "data_type": "query_result",
                            "fallback": True,
                            "original_error": str(sql_error),
                        }

            else:
                logger.info(f"[{self.name}] No SQL query - using smart fallback for complex query")

                query_lower = state["user_query"].lower()

                if "category" in query_lower or "product" in query_lower:
                    fallback_df = self.data_processor.get_top_categories(10)
                    extracted_data = {
                        "query_result": fallback_df.to_dict("records"),
                        "row_count": len(fallback_df),
                        "columns": list(fallback_df.columns),
                        "data_type": "query_result",
                        "fallback": True,
                        "fallback_reason": "Complex query - showing top categories instead",
                    }

                elif "region" in query_lower or "state" in query_lower:
                    fallback_df = self.data_processor.get_regional_performance()
                    extracted_data = {
                        "query_result": fallback_df.head(10).to_dict("records"),
                        "row_count": min(10, len(fallback_df)),
                        "columns": list(fallback_df.columns),
                        "data_type": "query_result",
                        "fallback": True,
                        "fallback_reason": "Complex query - showing regional performance instead",
                    }

                else:
                    summary_stats = self.data_processor.get_summary_statistics()
                    extracted_data = {
                        "query_result": [summary_stats.get("amazon_sales", {})],
                        "row_count": 1,
                        "columns": list(summary_stats.get("amazon_sales", {}).keys()),
                        "data_type": "query_result",
                        "fallback": True,
                        "fallback_reason": "Complex query - showing summary statistics instead",
                    }

            state["extracted_data"] = extracted_data
            state["metadata"]["data_extraction"] = {
                "success": True,
                "records_extracted": len(
                    extracted_data.get(
                        "query_result",
                        extracted_data.get("top_categories", []),
                    )
                ),
                "fallback_used": extracted_data.get("fallback", False),
            }

            logger.info(f"[{self.name}] Data extracted successfully")

        except Exception as e:
            logger.error(f"[{self.name}] Error: {str(e)}")
            state["errors"].append(f"Data Extraction Error: {str(e)}")
            state["metadata"]["data_extraction"] = {
                "success": False,
                "error": str(e),
            }
            state["extracted_data"] = {
                "data_type": "error",
                "error": str(e),
            }

        return state


class ValidationAgent:
    """
    Agent 3: Validates extracted data and ensures quality.
    """

    def __init__(self, llm: LLMProtocol):
        self.llm = llm
        self.name = "ValidationAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Validate extracted data and check for inconsistencies.
        """
        logger.info(f"[{self.name}] Validating data...")

        try:
            validation_result = {
                "is_valid": True,
                "confidence": 1.0,
                "issues": [],
                "recommendations": [],
            }

            extracted_data = state.get("extracted_data", {})

            if not extracted_data or extracted_data.get("data_type") == "error":
                validation_result["is_valid"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append("No data extracted or extraction failed")

            if state["query_type"] == "summarization":
                required_keys = ["summary_statistics", "top_categories", "regional_performance"]
                missing_keys = [key for key in required_keys if key not in extracted_data]

                if missing_keys:
                    validation_result["issues"].append(f"Missing data components: {missing_keys}")
                    validation_result["confidence"] *= 0.8

                if not extracted_data.get("summary_statistics"):
                    validation_result["issues"].append("Summary statistics are empty")
                    validation_result["confidence"] *= 0.8

            elif state["query_type"] == "qa":
                if "query_result" in extracted_data and len(extracted_data["query_result"]) == 0:
                    validation_result["recommendations"].append(
                        "Query returned no results. Consider broadening search criteria."
                    )
                    validation_result["confidence"] *= 0.7

            if extracted_data and extracted_data.get("data_type") != "error":
                rows = extracted_data.get("query_result") or extracted_data.get("top_categories") or []
                row_count = extracted_data.get(
                    "row_count",
                    len(rows) if isinstance(rows, list) else 0,
                )

                validation_result["llm_assessment"] = (
                    f"Rule-based validation passed. Extracted data contains {row_count} relevant record(s) "
                    f"for query type '{state['query_type']}'."
                )

            state["validation_result"] = validation_result
            state["metadata"]["validation"] = {
                "success": True,
                "is_valid": validation_result["is_valid"],
                "confidence": validation_result["confidence"],
            }

            logger.info(f"[{self.name}] Validation complete - Valid: {validation_result['is_valid']}")

        except Exception as e:
            logger.error(f"[{self.name}] Error: {str(e)}")
            state["errors"].append(f"Validation Error: {str(e)}")
            state["metadata"]["validation"] = {
                "success": False,
                "error": str(e),
            }
            state["validation_result"] = {
                "is_valid": True,
                "confidence": 0.5,
                "issues": [str(e)],
            }

        return state


class ResponseGenerationAgent:
    """
    Agent 4: Generates natural language response from validated data.
    """

    def __init__(self, llm: LLMProtocol):
        self.llm = llm
        self.name = "ResponseGenerationAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Generate human-readable response.
        """
        logger.info(f"[{self.name}] Generating response...")

        try:
            extracted_data = state.get("extracted_data", {})
            validation = state.get("validation_result") or {
                "is_valid": True,
                "confidence": 1.0,
                "issues": [],
            }

            is_fallback = extracted_data.get("fallback", False)
            has_error = extracted_data.get("data_type") == "error"
            empty_result = extracted_data.get("empty_result", False)
            dataset_metadata = extracted_data.get("dataset_metadata", {})

            conv_history = state.get("conversation_history", [])
            conv_context = ""

            if conv_history:
                recent = conv_history[-3:]
                turns = []

                for ex in recent:
                    resp = ex.get("response", "")
                    if len(resp) > 300:
                        resp = resp[:300] + "..."
                    turns.append(f"  User: {ex.get('query', '')}\n  Assistant: {resp}")

                conv_context = "\n\nCONVERSATION HISTORY for follow-up context:\n" + "\n".join(turns)

            system_prompt = f"""You are a Retail Analytics Expert creating insights for business executives.

Your task is to generate clear, concise, and actionable insights based on the extracted data.

Guidelines:
1. Start with a direct answer to the user's question.
2. Include specific numbers from the data.
3. Highlight key trends or patterns.
4. Use business-friendly language.
5. Keep Q&A responses to 3-5 sentences.
6. Keep summaries to 1-2 strong paragraphs.
7. If query returned empty results, explain why using dataset metadata.
8. If the exact query could not be answered, provide the most relevant alternative insights.
9. Always provide what you CAN show from the data.
10. If this is a follow-up question, maintain context from conversation history.
11. Never say "no data was provided" when Data to Present contains rows, summary_statistics, top_categories, or regional_performance.
{conv_context}

Format your response in a professional, easy-to-read manner."""

            fallback_note = ""

            if empty_result:
                date_range = dataset_metadata.get("date_range", {})
                user_query_lower = state["user_query"].lower()

                reasons = []

                if "yoy" in user_query_lower or "year-over-year" in user_query_lower or "year over year" in user_query_lower:
                    if date_range.get("unique_years", 0) < 2:
                        reasons.append(
                            f"YoY comparison requires at least 2 years of data, "
                            f"but dataset only contains {date_range.get('available_years', 'limited')} data"
                        )

                if "q3" in user_query_lower:
                    available_quarters = date_range.get("available_quarters", "")
                    if "Q3" not in available_quarters:
                        reasons.append(
                            f"Q3 data requested but not available. Dataset contains: {available_quarters}"
                        )

                if "q4" in user_query_lower:
                    available_quarters = date_range.get("available_quarters", "")
                    if "Q4" not in available_quarters:
                        reasons.append(
                            f"Q4 data requested but not available. Dataset contains: {available_quarters}"
                        )

                if reasons:
                    fallback_note = "\nDATA LIMITATION - Query returned empty because:\n"
                    fallback_note += "\n".join(f"- {r}" for r in reasons)
                    fallback_note += f"\n\nDataset coverage: {date_range.get('min_date', 'Unknown')} to {date_range.get('max_date', 'Unknown')}"
                    fallback_note += f"\nAvailable periods: {date_range.get('available_quarters', 'Unknown')}"
                    fallback_note += f"\n\nInstead, showing relevant insights from available data ({date_range.get('available_quarters', 'available periods')})."
                else:
                    fallback_note = "\nNOTE: Query returned no results. This might be due to specific filters not matching any data."

            elif is_fallback:
                fallback_note = "\nNOTE: The original query could not be executed as specified. Providing alternative relevant data instead."

            elif has_error:
                fallback_note = "\nNOTE: Data extraction encountered an error. Provide the best possible response acknowledging the limitation."

            # IMPORTANT FIX:
            # Summary mode does not store results in query_result.
            # It stores results in summary_statistics, top_categories, and regional_performance.
            # Passing only query_result makes Gemini think no data was provided.
            if extracted_data.get("data_type") == "summary":
                data_to_show = {
                    "summary_statistics": extracted_data.get("summary_statistics", {}),
                    "top_categories": extracted_data.get("top_categories", [])[:10],
                    "regional_performance": extracted_data.get("regional_performance", [])[:10],
                }
            else:
                data_to_show = extracted_data.get("query_result", [])
                if empty_result and extracted_data.get("fallback_data"):
                    data_to_show = extracted_data.get("fallback_data", [])

            user_prompt = f"""User Query: {state['user_query']}

Query Type: {state['query_type']}
{fallback_note}

Data to Present:
{json.dumps(data_to_show, indent=2, default=str)[:5000]}

Validation Status:
- Valid: {validation.get('is_valid', True)}
- Confidence: {validation.get('confidence', 1.0)}
- Issues: {validation.get('issues', [])}

Generate a clear, professional response using the numbers in Data to Present.

Important:
- Do NOT say no data was provided if Data to Present contains summary_statistics, top_categories, regional_performance, or rows.
- For summary queries, mention total sales/revenue if available, top categories, and regional performance.
- Start by explaining a data limitation only when the data is empty or explicitly marked as limited.
- Provide actionable business insights from the available data."""

            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            state["final_response"] = response.content
            state["metadata"]["response_generation"] = {
                "success": True,
            }

            logger.info(f"[{self.name}] Response generated successfully")

        except Exception as e:
            error_text = str(e)
            logger.error(f"[{self.name}] Error: {error_text}")

            state["final_response"] = _local_response_from_data(state, error_text)
            state["metadata"]["response_generation"] = {
                "success": False,
                "fallback_used": True,
                "error": error_text,
            }

            if not _is_quota_or_key_error(error_text):
                state["errors"].append(f"Response Generation Error: {error_text}")

        return state