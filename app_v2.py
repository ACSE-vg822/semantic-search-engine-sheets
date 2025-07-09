import streamlit as st
import json
import logging
from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced
from src.semantic_search.search_engine_v2 import SearchEngineV2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_v2_search_results(results):
    """Display the SearchEngineV2 results in Streamlit."""
    if results.get("status") != "success":
        st.error("❌ Search failed")
        return
    
    enriched_data = results.get("enriched_data", [])
    
    # Show detailed findings
    if enriched_data:
        st.markdown("### 🔍 Detailed Findings")
        
        for i, data in enumerate(enriched_data, 1):
            with st.expander(f"📌 {data['first_cell_value']} | {data['sheet']}"):
                st.markdown(f"**📋 Sheet:** `{data['sheet']}`")
                st.markdown(f"**📍 Location:** `{data['cell_addresses']}`")
                st.markdown(f"**📊 Values:** {data['values']}")
                
                if data.get('formulas'):
                    st.markdown(f"**🧮 Formulas:** `{', '.join(data['formulas'])}`")
                
                if data.get('cross_references'):
                    st.markdown("**🔗 Cross References:**")
                    for ref, value in data['cross_references'].items():
                        st.markdown(f"  - `{ref}`: {value}")

def build_and_cache_v2_search_engine(sheet_id, sheet_name):
    """Build and cache the SearchEngineV2 for the selected spreadsheet."""
    try:
        with st.spinner("📊 Building knowledge graph..."):
            parser = SpreadsheetParserAdvanced()
            spreadsheet_data = parser.parse(sheet_id)
            knowledge_graph = parser.build_knowledge_graph(spreadsheet_data)
            
            # Cache in session state
            st.session_state.knowledge_graph = knowledge_graph
            st.session_state.current_sheet_id = sheet_id
            st.session_state.current_sheet_name = sheet_name
            
        with st.spinner("🚀 Initializing SearchEngineV2..."):
            search_engine = SearchEngineV2(knowledge_graph, sheet_id, debug=False)  # Clean mode for UI
            st.session_state.search_engine_v2 = search_engine
            
        st.success(f"✅ SearchEngineV2 ready for: {knowledge_graph.title}")
        return search_engine
    except Exception as e:
        st.error(f"❌ Error building SearchEngineV2: {str(e)}")
        logger.error(f"Error building SearchEngineV2: {e}", exc_info=True)
        return None

# Hardcoded spreadsheet titles and IDs
SPREADSHEETS = {
    "Sales Dashboard": "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls",
    "Financial Model": "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc"
}

def extract_sheet_id_from_url(url_or_id):
    """Extract Google Sheets ID from URL or return the ID if already provided."""
    import re
    
    # If it's already just an ID (no slashes), return as-is
    if '/' not in url_or_id:
        return url_or_id.strip()
    
    # Extract from full Google Sheets URL
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    
    # Try alternative pattern for sharing URLs
    pattern = r'id=([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    
    # If no pattern matches, assume it's a direct ID
    return url_or_id.strip()

st.set_page_config(page_title="Semantic Search Engine V2", layout="wide")
st.title("🔍 Semantic Search Engine V2")
st.markdown("*Powered by LangGraph pipeline with pandas analysis and Claude insights*")

# Initialize session state
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'current_sheet_id' not in st.session_state:
    st.session_state.current_sheet_id = None
if 'current_sheet_name' not in st.session_state:
    st.session_state.current_sheet_name = None
if 'search_engine_v2' not in st.session_state:
    st.session_state.search_engine_v2 = None
if 'using_custom_sheet' not in st.session_state:
    st.session_state.using_custom_sheet = False
if 'custom_sheet_url' not in st.session_state:
    st.session_state.custom_sheet_url = ""

# Spreadsheet selector section
st.markdown("## 📊 Select Spreadsheet")

# Create three columns for different input methods with OR divider
col1, col_or, col2 = st.columns([5, 1, 5])

with col1:
    st.markdown("### 📋 Predefined Test Spreadsheets")
    spreadsheet_options = ["Please select a spreadsheet..."] + list(SPREADSHEETS.keys())
    sheet_choice = st.selectbox("Choose a spreadsheet:", spreadsheet_options, key="predefined_sheet")

with col_or:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 20px; font-weight: bold; color: #666; margin-top: 60px;'>OR</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### ➕ Add Your Own Spreadsheet")
    custom_url = st.text_input("Google Sheets URL or ID:", 
                               value=st.session_state.custom_sheet_url,
                               placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/... or just the ID", 
                               key="custom_url")
    
    col2a, col2b = st.columns([1, 1])
    with col2a:
        use_custom = st.button("Use Custom Spreadsheet", key="use_custom_btn")
    with col2b:
        if st.session_state.using_custom_sheet:
            clear_custom = st.button("🗑️ Clear Custom", key="clear_custom_btn", type="primary")
            if clear_custom:
                st.session_state.using_custom_sheet = False
                st.session_state.custom_sheet_url = ""
                st.session_state.knowledge_graph = None
                st.session_state.current_sheet_id = None
                st.session_state.current_sheet_name = None
                st.session_state.search_engine_v2 = None
                st.success("✅ Custom spreadsheet cleared!")

# Handle custom spreadsheet button click
if use_custom and custom_url.strip():
    st.session_state.using_custom_sheet = True
    st.session_state.custom_sheet_url = custom_url.strip()
    st.session_state.knowledge_graph = None
    st.session_state.current_sheet_id = None
    st.session_state.current_sheet_name = None
    st.session_state.search_engine_v2 = None
    st.success(f"✅ Custom spreadsheet selected!")
elif use_custom and not custom_url.strip():
    st.error("❌ Please provide a Google Sheets URL or ID for your custom spreadsheet.")

# Handle predefined spreadsheet selection
if 'previous_predefined_choice' not in st.session_state:
    st.session_state.previous_predefined_choice = "Please select a spreadsheet..."

if (sheet_choice != st.session_state.previous_predefined_choice and 
    sheet_choice != "Please select a spreadsheet..." and 
    not use_custom):
    
    if st.session_state.using_custom_sheet:
        st.session_state.knowledge_graph = None
        st.session_state.current_sheet_id = None
        st.session_state.current_sheet_name = None
        st.session_state.search_engine_v2 = None
    st.session_state.using_custom_sheet = False
    st.session_state.custom_sheet_url = ""
    st.session_state.previous_predefined_choice = sheet_choice

# Determine which spreadsheet to use
sheet_id = None
sheet_name = None

if st.session_state.using_custom_sheet and st.session_state.custom_sheet_url:
    try:
        sheet_id = extract_sheet_id_from_url(st.session_state.custom_sheet_url)
        sheet_name = "Custom Spreadsheet"
        st.info(f"📋 Using custom spreadsheet with ID: {sheet_id}")
    except Exception as e:
        st.error(f"❌ Error processing custom spreadsheet URL/ID: {str(e)}")
        st.session_state.using_custom_sheet = False
        st.session_state.custom_sheet_url = ""
        sheet_id = None
        sheet_name = None
elif not st.session_state.using_custom_sheet and sheet_choice != "Please select a spreadsheet...":
    sheet_id = SPREADSHEETS[sheet_choice]
    sheet_name = sheet_choice

# Build search engine if needed
if not sheet_id:
    search_engine = None
    st.info("👆 Please select a predefined spreadsheet or add your own custom spreadsheet to begin")
else:
    # Check if user selected a different spreadsheet
    if st.session_state.current_sheet_id != sheet_id:
        search_engine = build_and_cache_v2_search_engine(sheet_id, sheet_name)
    else:
        search_engine = st.session_state.search_engine_v2
        if search_engine:
            st.info(f"📋 Using cached SearchEngineV2 for: {st.session_state.knowledge_graph.title}")

# Show spreadsheet overview if knowledge graph is available
if st.session_state.knowledge_graph:
    with st.expander("📈 Spreadsheet Overview"):
        kg = st.session_state.knowledge_graph
        st.write(f"**Title:** {kg.title}")
        st.write(f"**Sheets:** {len(kg.sheets)}")
        for sheet_name_overview, sheet_meta in kg.sheets.items():
            st.write(f"  - **{sheet_name_overview}**: {len(sheet_meta.columns)} columns")

# Only show search functionality if search engine is ready
if search_engine:
    # Initialize query state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Query input
    st.markdown("## 🔍 Advanced Semantic Search")
    
    # Sample questions
    st.markdown("### 💡 Try these sample questions:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💰 Find all revenue calculations", key="sample1"):
            st.session_state.current_query = "Find all revenue calculations"
        if st.button("🔧 Show me cost-related formulas", key="sample2"):
            st.session_state.current_query = "Show me cost-related formulas"
    
    with col2:
        if st.button("📊 Where are my margin analyses?", key="sample3"):
            st.session_state.current_query = "Where are my margin analyses?"
        if st.button("📈 What percentage calculations do I have?", key="sample4"):
            st.session_state.current_query = "What percentage calculations do I have?"
    
    # Query input with current query as value
    query = st.text_input("Enter your semantic query:", value=st.session_state.current_query)
    
    # Update current query when user types
    if query != st.session_state.current_query:
        st.session_state.current_query = query
    
    if st.button("🚀 Search with V2 Engine") and query.strip():
        try:
            with st.spinner("🧠 Running LangGraph pipeline..."):
                # Use the cached SearchEngineV2
                results = search_engine.search(query)
                
            st.success("🎯 Search Complete!")
            
            # Display results using the new format
            display_v2_search_results(results)
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Error in SearchEngineV2: {e}", exc_info=True)
            
            # Show more details for debugging
            with st.expander("🔍 Error Details"):
                st.code(str(e)) 