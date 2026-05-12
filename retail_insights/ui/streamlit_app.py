"""
Streamlit UI for Retail Insights Assistant.
Clean full-width interface for sales analytics powered by the existing agent workflow.
"""
from __future__ import annotations

from datetime import datetime
import os

from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st

from retail_insights.orchestrator import RetailInsightsOrchestrator


load_dotenv(override=True)

st.set_page_config(
    page_title="Retail Insights Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXAMPLE_QUESTIONS = [
    "Which category has the highest sales?",
    "What is the total revenue by state?",
    "Show me top performing products",
    "Which region saw the most orders?",
    "What are the sales trends?",
]

st.markdown(
    """
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --panel: #ffffff;
        --soft: #f8fafc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .hero-card {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 55%, #f8fafc 100%);
        border: 1px solid #dbeafe;
        border-radius: 28px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
    }

    .main-header {
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 850;
        line-height: 1.05;
        color: var(--ink);
        letter-spacing: -0.04em;
        margin: 0;
    }

    .sub-header {
        font-size: 1rem;
        color: var(--muted);
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .top-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.48rem 0.85rem;
        color: #334155;
        font-size: 0.9rem;
        font-weight: 650;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    .section-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        margin: 0.75rem 0 1rem;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
    }

    .chat-shell {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 26px;
        padding: 1rem 1.1rem 1.2rem;
        box-shadow: 0 16px 44px rgba(15, 23, 42, 0.06);
    }

    .chat-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.95rem;
    }

    .insight-box {
        background: #ffffff;
        padding: 1.2rem;
        border-left: 5px solid var(--primary);
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        color: #111827;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 999px !important;
        font-weight: 750 !important;
        border: 1px solid #cbd5e1 !important;
        min-height: 2.75rem;
        transition: all 0.18s ease-in-out;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: 0 !important;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 999px;
        min-height: 2.75rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.15rem 0.25rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Get API key from session first, then .env/environment."""
    session_key = st.session_state.get("gemini_api_key", "")
    if session_key and session_key.strip():
        return session_key.strip()

    env_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if env_key:
        return env_key

    return None


def get_api_key_source() -> str:
    """Return where the active key came from."""
    session_key = st.session_state.get("gemini_api_key", "")
    if session_key and session_key.strip():
        return "UI session"

    env_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if env_key:
        return ".env / environment"

    return "none"


@st.cache_resource
def initialize_orchestrator(api_key: str, model_name: str, data_path: str, temperature: float):
    """Initialize the existing multi-agent orchestrator."""
    return RetailInsightsOrchestrator(
        api_key=api_key,
        model_name=model_name,
        data_path=data_path,
        temperature=temperature,
    )


def select_example_question(question: str) -> None:
    """Prefill the chat input with an example question."""
    st.session_state.user_query_input = question


def render_header() -> None:
    """Render the clean header with top-level actions."""
    api_key = get_api_key()
    key_source = get_api_key_source()
    key_status = "Key active" if api_key else "Key required"
    key_icon = "✅" if api_key else "🔑"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="main-header">📊 Retail Insights Assistant</div>
            <p class="sub-header">Ask business questions, generate summaries, and inspect sales.</p>
            <div style="margin-top: 1rem; display: flex; gap: 0.65rem; flex-wrap: wrap;">
                <span class="top-pill">{key_icon} {key_status} · {key_source}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_col1, action_col2, action_col3 = st.columns([1.2, 1.2, 5])

    with action_col1:
        with st.popover("🔍 View Questions", use_container_width=True):
            st.markdown("**Try these questions**")
            for question in EXAMPLE_QUESTIONS:
                st.button(
                    question,
                    key=f"example_{question}",
                    use_container_width=True,
                    on_click=select_example_question,
                    args=(question,),
                )


def render_api_key_manager() -> str | None:
    """Render API key controls at the top of the page, not in a sidebar."""
    api_key = get_api_key()
    key_source = get_api_key_source()

    if "show_api_key_input" not in st.session_state:
        st.session_state.show_api_key_input = not bool(api_key)

    show_input = st.session_state.show_api_key_input or not api_key

    with st.expander("🔑 Gemini API Key", expanded=show_input):
        if api_key and not show_input:
            masked = api_key[:4] + "..." + api_key[-4:]
            st.success(f"API key active from {key_source}: `{masked}`")
            if st.button("Change API Key", key="change_api_key_inside", use_container_width=True):
                st.session_state.show_api_key_input = True
                st.session_state.api_key_input = ""
                st.rerun()
        else:
            with st.form("api_key_form", clear_on_submit=False):
                key_input = st.text_input(
                    "Enter Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    key="api_key_input",
                )
                submitted = st.form_submit_button(
                    "Activate Key",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if key_input and len(key_input.strip()) > 10:
                    st.session_state.gemini_api_key = key_input.strip()
                    st.session_state.show_api_key_input = False
                    initialize_orchestrator.clear()
                    st.rerun()
                else:
                    st.error("Please enter a valid Gemini API key.")

            st.caption("Session-only API key override")

    return get_api_key()


def display_summary_visualizations(data: dict) -> None:
    """Display visualizations for summary data."""
    if not data or data.get("data_type") != "summary":
        return

    if "top_categories" in data and data["top_categories"]:
        st.markdown("### 📈 Top Performing Categories")
        df_categories = pd.DataFrame(data["top_categories"])

        if not df_categories.empty and "Category" in df_categories.columns:
            fig = px.bar(
                df_categories.head(10),
                x="Category",
                y="total_revenue",
                title="Top 10 Categories by Revenue",
                labels={"total_revenue": "Total Revenue (INR)", "Category": "Category"},
                color="total_revenue",
                color_continuous_scale="blues",
            )
            st.plotly_chart(fig, width="stretch")

    if "regional_performance" in data and data["regional_performance"]:
        st.markdown("### 🗺️ Regional Performance")
        df_regions = pd.DataFrame(data["regional_performance"])

        if not df_regions.empty and "state" in df_regions.columns:
            fig = px.bar(
                df_regions.head(10),
                x="state",
                y="total_revenue",
                title="Top 10 States by Revenue",
                labels={"total_revenue": "Total Revenue (INR)", "state": "State"},
                color="order_count",
                color_continuous_scale="viridis",
            )
            st.plotly_chart(fig, width="stretch")


def render_extracted_data_details(result: dict) -> None:
    """Render SQL, table, and chart details without agent status panels."""
    if result.get("sql_query"):
        st.markdown("### 🔍 Generated SQL Query")
        st.code(result["sql_query"], language="sql")
        st.markdown("---")

    if result.get("data"):
        st.markdown("### 📊 Extracted Data")
        data = result["data"]

        if data.get("data_type") == "query_result" and data.get("query_result"):
            df = pd.DataFrame(data["query_result"])

            tab1, tab2 = st.tabs(["📋 Table View", "📈 Chart View"])

            with tab1:
                st.dataframe(df, width="stretch")

            with tab2:
                try:
                    numeric_cols = df.select_dtypes(
                        include=["float64", "int64", "int32", "float32"]
                    ).columns.tolist()
                    categorical_cols = df.select_dtypes(
                        include=["object", "string"]
                    ).columns.tolist()
                    date_cols = [
                        col
                        for col in df.columns
                        if "date" in col.lower()
                        or "time" in col.lower()
                        or "month" in col.lower()
                        or "year" in col.lower()
                    ]

                    num_rows = len(df)
                    chart_created = False

                    if date_cols and numeric_cols:
                        date_col = date_cols[0]
                        num_col = numeric_cols[0]

                        fig = px.line(
                            df,
                            x=date_col,
                            y=num_col,
                            title=f"{num_col} Over Time",
                            labels={date_col: date_col, num_col: num_col},
                            markers=True,
                        )
                        fig.update_layout(hovermode="x unified")
                        st.plotly_chart(fig, width="stretch")
                        chart_created = True

                    elif len(categorical_cols) > 0 and len(numeric_cols) > 0 and 2 <= num_rows <= 8:
                        cat_col = categorical_cols[0]
                        num_col = numeric_cols[0]

                        fig = px.pie(
                            df,
                            names=cat_col,
                            values=num_col,
                            title=f"{num_col} Distribution by {cat_col}",
                            hole=0.3,
                        )
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                        st.plotly_chart(fig, width="stretch")
                        chart_created = True

                    elif len(categorical_cols) > 0 and len(numeric_cols) > 0 and num_rows > 8:
                        cat_col = categorical_cols[0]
                        num_col = numeric_cols[0]
                        df_sorted = df.nlargest(15, num_col)

                        fig = px.bar(
                            df_sorted,
                            y=cat_col,
                            x=num_col,
                            orientation="h",
                            title=f"Top 15: {num_col} by {cat_col}",
                            labels={cat_col: cat_col, num_col: num_col},
                            color=num_col,
                            color_continuous_scale="blues",
                        )
                        st.plotly_chart(fig, width="stretch")
                        chart_created = True

                    elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        cat_col = categorical_cols[0]
                        num_col = numeric_cols[0]

                        fig = px.bar(
                            df.head(10),
                            x=cat_col,
                            y=num_col,
                            title=f"{num_col} by {cat_col}",
                            labels={cat_col: cat_col, num_col: num_col},
                            color=num_col,
                            color_continuous_scale="viridis",
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, width="stretch")
                        chart_created = True

                    elif len(numeric_cols) >= 2:
                        first_col = df.columns[0]

                        fig = px.bar(
                            df.head(10),
                            x=first_col,
                            y=numeric_cols[:3],
                            title="Multi-Metric Comparison",
                            barmode="group",
                            labels={first_col: first_col},
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, width="stretch")
                        chart_created = True

                    elif len(numeric_cols) == 2 and len(categorical_cols) > 0:
                        cat_col = categorical_cols[0]

                        fig = px.scatter(
                            df.head(50),
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color=cat_col,
                            size=numeric_cols[0],
                            title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                            labels={
                                numeric_cols[0]: numeric_cols[0],
                                numeric_cols[1]: numeric_cols[1],
                            },
                        )
                        st.plotly_chart(fig, width="stretch")
                        chart_created = True

                    elif len(numeric_cols) == 1 and num_rows == 1:
                        num_col = numeric_cols[0]
                        value = df[num_col].iloc[0]

                        st.metric(
                            label=num_col.replace("_", " ").title(),
                            value=f"{value:,.2f}" if isinstance(value, (int, float)) else str(value),
                        )
                        chart_created = True

                    if not chart_created:
                        st.info("📊 No suitable visualization for this data type. View the table for details.")

                except Exception as exc:
                    st.warning(f"Could not create visualization: {str(exc)}")
                    st.info("💡 Tip: Visualization works best with numeric data.")


def render_chat_workspace(orchestrator: RetailInsightsOrchestrator) -> None:
    """Render the full-width chat workspace."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_query_input" not in st.session_state:
        st.session_state.user_query_input = ""

    st.markdown(
        """
        <div class="section-card">
            <div class="chat-title">💬 Chat with Your Sales Data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(chat["query"])
                    st.caption(chat["timestamp"])

                with st.chat_message("assistant"):
                    st.markdown(chat["result"].get("response", ""))
                    with st.expander("🔍 View Details"):
                        render_extracted_data_details(chat["result"])
        else:
            st.info("👋 Query your retail data naturally below, or browse **View Questions** for example insights.")

    input_col, clear_col = st.columns([5, 1])

    with input_col:
        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask a question",
                placeholder="e.g., Which category saw the highest sales?",
                key="user_query_input",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(
                "Ask Me",
                type="primary",
                use_container_width=True,
            )

    with clear_col:
        st.write("")
        st.write("")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if submit_button and user_query.strip():
        try:
            conv_history = [
                {
                    "query": chat["query"],
                    "response": chat["result"].get("response", ""),
                }
                for chat in st.session_state.chat_history
            ]

            with st.spinner("Analyzing your sales data..."):
                result = orchestrator.process_query(
                    user_query.strip(),
                    query_type="qa",
                    conversation_history=conv_history,
                )

        except Exception as exc:
            st.error(f"❌ Error during processing: {str(exc)}")
            st.stop()

        st.session_state.chat_history.append(
            {
                "query": user_query.strip(),
                "result": result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.rerun()


def render_summary_workspace(orchestrator: RetailInsightsOrchestrator) -> None:
    """Render the full-width summary workspace."""
    st.markdown(
        """
        <div class="section-card">
            <div class="chat-title">📋 Generate Comprehensive Sales Summary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Generate Business Summary →", type="primary", use_container_width=True):
        try:
            with st.spinner("Preparing sales summary..."):
                result = orchestrator.generate_summary()
        except Exception as exc:
            st.error(f"❌ Error during processing: {str(exc)}")
            st.stop()

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Executive Summary")
        st.markdown(result["response"])
        st.markdown("</div>", unsafe_allow_html=True)

        if result.get("data"):
            st.markdown("---")
            display_summary_visualizations(result["data"])

            with st.expander("📊 View Detailed Statistics"):
                summary_stats = result["data"].get("summary_statistics", {})

                if "amazon_sales" in summary_stats:
                    st.markdown("#### Amazon Sales Statistics")
                    amazon_stats = summary_stats["amazon_sales"]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Orders", f"{amazon_stats.get('total_orders', 0):,}")

                    with col2:
                        st.metric("Total Revenue", f"₹{amazon_stats.get('total_revenue', 0):,.2f}")

                    with col3:
                        st.metric("Avg Order Value", f"₹{amazon_stats.get('avg_order_value', 0):,.2f}")

                if "international_sales" in summary_stats:
                    st.markdown("#### International Sales Statistics")
                    intl_stats = summary_stats["international_sales"]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Transactions", f"{intl_stats.get('total_transactions', 0):,}")

                    with col2:
                        st.metric("Total Revenue", f"₹{intl_stats.get('total_revenue', 0):,.2f}")

                    with col3:
                        st.metric("Unique Customers", f"{intl_stats.get('unique_customers', 0):,}")


def main() -> None:
    """Main Streamlit application."""
    render_header()
    api_key = render_api_key_manager()

    if not api_key:
        st.warning("🔑 Add your Gemini API key above to start using the app.")
        st.stop()

    try:
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
        data_path = os.getenv("DATA_PATH", "Sales Dataset/")
        temperature = float(os.getenv("TEMPERATURE", "0.1"))

        orchestrator = initialize_orchestrator(
            api_key,
            model_name,
            data_path,
            temperature,
        )
    except Exception as exc:
        st.error(f"Failed to initialize system: {str(exc)}")
        st.stop()

    st.markdown(
        """
        <div class="section-card">
            <div class="small-muted">Choose Q&A for direct questions or Summary for a complete business overview.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.segmented_control(
        "Choose mode",
        ["💬 Conversational Q&A", "📋 Generate Summary"],
        default="💬 Conversational Q&A",
        label_visibility="collapsed",
    )

    st.markdown("---")

    if mode == "💬 Conversational Q&A":
        render_chat_workspace(orchestrator)
    else:
        render_summary_workspace(orchestrator)


if __name__ == "__main__":
    main()
