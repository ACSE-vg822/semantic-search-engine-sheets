import streamlit as st
import json
import logging
from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced
from src.rag.retriever import SpreadsheetRetriever
from src.semantic_search.search_engine_advanced import QueryEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_claude_response(response):
    """Parse JSON from Claude's response and return parsed results."""
    try:
        # Extract JSON from Claude's response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end != 0:
            json_text = response[json_start:json_end]
            parsed_results = json.loads(json_text)
        else:
            # Fallback: try to parse the entire response
            parsed_results = json.loads(response)
        return parsed_results, None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON response from Claude: {str(e)}"
    except Exception as e:
        return None, f"Error processing results: {str(e)}"

def normalize_formula_display(formula):
    """Normalize formula for display by replacing numbers with subscript n."""
    if not formula:
        return formula
    
    import re
    # Replace numbers in cell references with subscript n
    # Pattern matches letter(s) followed by number(s)
    normalized = re.sub(r'([A-Z]+)(\d+)', r'\1<sub>n</sub>', formula)
    return normalized

def display_search_results(parsed_results):
    """Display the parsed search results in Streamlit."""
    for i, res in enumerate(parsed_results, 1):
        with st.expander(f"📌 Result {i}: {res.get('concept_name', 'N/A')} in sheet '{res.get('sheet', 'N/A')}'"):
            st.markdown(f"**Header:** `{res.get('header', 'N/A')}`")
            st.markdown(f"**Cell Range:** `{res.get('cell_range', 'N/A')}`")
            if res.get('formula'):
                normalized_formula = normalize_formula_display(res['formula'])
                st.markdown(f"**Formula:** {normalized_formula}", unsafe_allow_html=True)
            else:
                st.markdown("*No formula*")
            if res.get('explanation'):
                st.markdown(f"**Explanation:** {res['explanation']}")
            
            # Show cross-sheet references if available
            if 'cross_sheet_refs' in res and res['cross_sheet_refs']:
                st.markdown(f"**Cross-sheet references:** {', '.join(res['cross_sheet_refs'])}")

def build_and_cache_knowledge_graph(sheet_id, sheet_name):
    """Build and cache the knowledge graph for the selected spreadsheet."""
    try:
        with st.spinner("📊 Building knowledge graph for faster queries..."):
            parser = SpreadsheetParserAdvanced()
            spreadsheet_data = parser.parse(sheet_id)
            knowledge_graph = parser.build_knowledge_graph(spreadsheet_data)
            
            # Cache in session state
            st.session_state.knowledge_graph = knowledge_graph
            st.session_state.current_sheet_id = sheet_id
            st.session_state.current_sheet_name = sheet_name
            
        st.success(f"✅ Knowledge graph built for: {knowledge_graph.title}")
        return knowledge_graph
    except Exception as e:
        st.error(f"❌ Error building knowledge graph: {str(e)}")
        logger.error(f"Error building knowledge graph: {e}", exc_info=True)
        return None

# Hardcoded spreadsheet titles and IDs
SPREADSHEETS = {
    "Sales Dashboard": "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls",
    "Financial Model": "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc"
}

st.set_page_config(page_title="Advanced Semantic Spreadsheet Search", layout="wide")
st.title("🔍 Semantic Spreadsheet Search")
st.markdown("*Powered by enhanced knowledge graphs and statistical analysis*")

# Initialize session state
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'current_sheet_id' not in st.session_state:
    st.session_state.current_sheet_id = None
if 'current_sheet_name' not in st.session_state:
    st.session_state.current_sheet_name = None

# Spreadsheet selector
spreadsheet_options = ["Please select a spreadsheet..."] + list(SPREADSHEETS.keys())
sheet_choice = st.selectbox("Choose a spreadsheet:", spreadsheet_options)

# Only proceed if a valid spreadsheet is selected
if sheet_choice == "Please select a spreadsheet...":
    knowledge_graph = None
    st.info("👆 Please select a spreadsheet to begin")
else:
    sheet_id = SPREADSHEETS[sheet_choice]
    
    # Check if user selected a different spreadsheet and build/cache knowledge graph
    if st.session_state.current_sheet_id != sheet_id:
        knowledge_graph = build_and_cache_knowledge_graph(sheet_id, sheet_choice)
    else:
        knowledge_graph = st.session_state.knowledge_graph
        if knowledge_graph:
            st.info(f"📋 Using cached knowledge graph for: {knowledge_graph.title}")

# Show spreadsheet overview if knowledge graph is available
if knowledge_graph:
    with st.expander("📈 Spreadsheet Overview"):
        st.write(f"**Title:** {knowledge_graph.title}")
        st.write(f"**Sheets:** {len(knowledge_graph.sheets)}")
        for sheet_name, sheet_meta in knowledge_graph.sheets.items():
            st.write(f"  - **{sheet_name}**: {len(sheet_meta.columns)} columns")

# Only show search functionality if a valid spreadsheet is selected
if sheet_choice != "Please select a spreadsheet...":
    # Query input
    query = st.text_input("Enter your semantic query:")
    
    if st.button("Search") and query.strip():
        if not knowledge_graph:
            st.error("❌ Please select a spreadsheet first.")
            st.stop()
            
        try:
            # Step 1: Create retriever with the cached knowledge graph
            with st.spinner("🧠 Setting up semantic retriever..."):
                retriever = SpreadsheetRetriever(
                    knowledge_graph, 
                    use_embeddings=True
                )
                
            # Step 2: Create query engine with the retriever
            with st.spinner("⚙️ Initializing query engine..."):
                engine = QueryEngine(retriever, spreadsheet_id=SPREADSHEETS[sheet_choice])
                
            # Step 3: Process the query
            with st.spinner("🔍 Processing your query with Claude..."):
                response = engine.ask(query)
                
            st.success("🎯 Search Results")
            
            # Parse and display Claude's response
            parsed_results, error_message = parse_claude_response(response)
            if error_message:
                st.error(error_message)
                st.write("**Raw Response:**")
                st.code(response, language="text")
            else:
                display_search_results(parsed_results)
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Error in main flow: {e}", exc_info=True)
        
# Footer with information about the advanced features
st.markdown("---")
st.markdown("""
### 🚀 Advanced Features:
- **Knowledge Graph Caching**: Builds and caches knowledge graph on spreadsheet selection for faster queries
- **Knowledge Graph**: Builds comprehensive metadata including cross-sheet references
- **Statistical Analysis**: Automatically calculates min/max/mean for numerical columns
- **Semantic Embeddings**: Uses sentence transformers for better semantic matching
- **Enhanced Context**: Provides Claude with rich statistical and structural information
""")