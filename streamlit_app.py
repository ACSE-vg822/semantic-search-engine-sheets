import streamlit as st
import json
import time
from typing import Dict, Any

from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced
from src.rag.retriever import SpreadsheetRetriever
from src.semantic_search.langgraph_search_engine import LangGraphSearchEngine


def initialize_search_engine(spreadsheet_id: str) -> tuple[LangGraphSearchEngine, str]:
    """Initialize the search engine with a given spreadsheet ID"""
    try:
        # Check if we already have cached components for this spreadsheet
        cache_key = f"kg_{spreadsheet_id}"
        retriever_key = f"retriever_{spreadsheet_id}"
        title_key = f"title_{spreadsheet_id}"
        
        # Try to get cached knowledge graph and retriever
        if cache_key in st.session_state and retriever_key in st.session_state:
            st.info("🚀 Using cached knowledge graph and retriever")
            kg = st.session_state[cache_key]
            retriever = st.session_state[retriever_key]
            spreadsheet_title = st.session_state[title_key]
        else:
            # Build new components if not cached
            with st.spinner("🔧 Parsing spreadsheet and building knowledge graph..."):
                parser = SpreadsheetParserAdvanced()
                spreadsheet = parser.parse(spreadsheet_id)
                kg = parser.build_knowledge_graph(spreadsheet)
                
            with st.spinner("🧠 Setting up RAG retriever..."):
                retriever = SpreadsheetRetriever(kg, debug=False)
                
            # Cache the components
            st.session_state[cache_key] = kg
            st.session_state[retriever_key] = retriever
            st.session_state[title_key] = spreadsheet.title
            spreadsheet_title = spreadsheet.title
            
        with st.spinner("🚀 Initializing LangGraph search engine..."):
            search_engine = LangGraphSearchEngine(retriever)
            
        return search_engine, spreadsheet_title
        
    except Exception as e:
        st.error(f"❌ Error initializing search engine: {str(e)}")
        return None, None


def display_search_results(result: Dict[str, Any]):
    """Display search results in a nice format with expandable cards"""
    
    # Show query classification
    st.subheader("🔍 Query Analysis")
    query_type = result.get("query_type", "unknown")
    if query_type == "search":
        st.success(f"✅ Query classified as: **{query_type.upper()}**")
    elif query_type == "calculate":
        st.info(f"🧮 Query classified as: **{query_type.upper()}**")
    else:
        st.warning(f"⚠️ Query classified as: **{query_type.upper()}**")
    
    # Show number of results
    num_results = result.get("num_results", 0)
    st.write(f"**Found {num_results} relevant result(s):**")
    
    # Display individual result cards
    st.subheader("📊 Search Results")
    
    # Parse the response to extract individual results
    response_text = result.get("response", "No response available")
    
    # Try to extract individual results from the response
    raw_results = result.get("raw_results", [])
    
    if raw_results and isinstance(raw_results, list):
        # Display each result as an expandable card
        for i, raw_result in enumerate(raw_results, 1):
            # Extract information using the correct field names from the actual data structure
            result_type = raw_result.get("type", "Unknown")
            result_name = raw_result.get("name", "Unknown")
            sheet_name = raw_result.get("sheet", "Unknown Sheet")
            range_info = raw_result.get("addresses", "")
            relevance_explanation = raw_result.get("relevance_explanation", "")
            sample_values = raw_result.get("sample_values", {})
            data_type = raw_result.get("data_type", "")
            score = raw_result.get("score", 0)
            
            # Create sample data string from sample_values
            sample_text = ""
            if sample_values and isinstance(sample_values, dict):
                sample_items = []
                for key, value in sample_values.items():
                    sample_items.append(f"{key}: {value}")
                sample_text = ", ".join(sample_items)
            
            # Create card header with key information
            if result_type == "row":
                card_title = f"📄 {result_name} • {sheet_name}"
            elif result_type == "column":
                card_title = f"📊 {result_name} • {sheet_name}"
            else:
                card_title = f"📋 {result_name} • {sheet_name}"
            
            # Add range info to title if available
            if range_info:
                card_title += f" • {range_info}"
            
            # Create expandable card
            with st.expander(f"{i}. {card_title}", expanded=False):
                # Create columns for better layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**📋 Details:**")
                    if result_name != "Unknown":
                        st.write(f"• **Name:** {result_name}")
                    if result_type:
                        st.write(f"• **Type:** {result_type.title()}")
                    if sheet_name != "Unknown Sheet":
                        st.write(f"• **Sheet:** {sheet_name}")
                    if range_info:
                        st.write(f"• **Range:** {range_info}")
                    if data_type:
                        st.write(f"• **Data Type:** {data_type}")
                    if score > 0:
                        st.write(f"• **Relevance Score:** {score:.3f}")
                    
                    if relevance_explanation:
                        st.markdown("**🎯 Why it's relevant:**")
                        st.write(relevance_explanation)
                
                with col2:
                    if sample_text:
                        st.markdown("**🔍 Sample Data:**")
                        st.code(sample_text, language="text")
                
                # Show full raw data in a collapsible section
                with st.expander("🔧 View Raw Data", expanded=False):
                    st.json(raw_result)
    else:
        # Fallback: display the formatted response as before
        st.markdown(response_text)
        
        # Show raw results in expandable section
        if result.get("raw_results"):
            with st.expander("🔍 View Raw Results (JSON)"):
                st.json(result["raw_results"])


def main():
    st.set_page_config(
        page_title="Semantic Search Engine for Spreadsheets",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Semantic Search Engine for Spreadsheets")
    st.markdown("Search through your Google Sheets using natural language with AI-powered analysis!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Spreadsheet ID input
        spreadsheet_id = st.text_input(
            "📊 Google Spreadsheet ID",
            value=st.session_state.get("spreadsheet_id", ""),
            placeholder="1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc",
            help="Enter the ID from your Google Sheets URL"
        )
        
        # Example spreadsheet IDs
        st.markdown("**📋 Example Spreadsheet IDs:**")
        example_ids = {
            "Financial Model": "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc",
            "Sales Dashboard": "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"
        }
        
        for name, id_val in example_ids.items():
            if st.button(f"📌 {name}", key=f"example_{name}"):
                st.session_state.spreadsheet_id = id_val
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Search Interface")
        
        # Query input
        query = st.text_input(
            "💬 Enter your query:",
            value=st.session_state.get("current_query", ""),
            placeholder="e.g., 'show me revenue data' or 'calculate total expenses'",
            help="Ask questions about your spreadsheet data"
        )
        
        # Example queries
        st.markdown("**💡 Example Queries:**")
        example_queries = [
            "show me revenue data",
            "what expenses do we have?",
            "find salary information", 
            "calculate total revenue",
            "what costs are tracked?"
        ]
        
        query_cols = st.columns(3)
        for i, example_query in enumerate(example_queries):
            col_idx = i % 3
            with query_cols[col_idx]:
                if st.button(f"💡 {example_query}", key=f"example_query_{i}"):
                    st.session_state.current_query = example_query
                    st.rerun()
        
    with col2:
        st.subheader("ℹ️ How it Works")
        st.markdown("""
        1. **🔍 Query Classification**: AI determines if you want to search or calculate
        2. **📊 RAG Retrieval**: Finds relevant data from your spreadsheet
        3. **🤖 AI Analysis**: Filters results and explains relevance
        """)
    
    # Search button and results
    if st.button("🚀 Search", type="primary", disabled=not (spreadsheet_id and query)):
        if not spreadsheet_id:
            st.error("❌ Please enter a spreadsheet ID")
            return
        if not query:
            st.error("❌ Please enter a search query")
            return
            
        # Initialize search engine (will use cached components if available)
        search_engine, spreadsheet_title = initialize_search_engine(spreadsheet_id)
        if search_engine is None:
            return
        
        # Display current spreadsheet info
        st.info(f"📊 **Spreadsheet:** {spreadsheet_title}")
        
        # Perform search
        try:
            with st.spinner(f"🔍 Processing query: '{query}'..."):
                start_time = time.time()
                result = search_engine.search(query)
                end_time = time.time()
                
            st.success(f"✅ Search completed in {end_time - start_time:.2f} seconds")
            
            # Display results
            display_search_results(result)
            
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("🔧 **Built with:** Streamlit, LangGraph, Claude AI, and Sentence Transformers")


if __name__ == "__main__":
    main() 