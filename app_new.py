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

# Hardcoded spreadsheet titles and IDs
SPREADSHEETS = {
    "Sales Dashboard": "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls",
    "Financial Model": "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc"
}

st.set_page_config(page_title="Advanced Semantic Spreadsheet Search", layout="wide")
st.title("🔍 Semantic Spreadsheet Search")
st.markdown("*Powered by enhanced knowledge graphs and statistical analysis*")

# Spreadsheet selector
sheet_choice = st.selectbox("Choose a spreadsheet:", list(SPREADSHEETS.keys()))
sheet_id = SPREADSHEETS[sheet_choice]

# Query input
query = st.text_input("Enter your semantic query:")

if st.button("Search") and query.strip():
    try:
        # Step 1: Parse the spreadsheet and build knowledge graph
        with st.spinner("📊 Parsing spreadsheet and building knowledge graph..."):
            parser = SpreadsheetParserAdvanced()
            spreadsheet_data = parser.parse(sheet_id)
            knowledge_graph = parser.build_knowledge_graph(spreadsheet_data)
            
        st.success(f"✅ Parsed spreadsheet: {knowledge_graph.title}")
        
        # Show spreadsheet overview
        with st.expander("📈 Spreadsheet Overview"):
            st.write(f"**Title:** {knowledge_graph.title}")
            st.write(f"**Sheets:** {len(knowledge_graph.sheets)}")
            for sheet_name, sheet_meta in knowledge_graph.sheets.items():
                st.write(f"  - **{sheet_name}**: {len(sheet_meta.columns)} columns")
        
        # Step 2: Create retriever with the knowledge graph
        with st.spinner("🧠 Setting up semantic retriever..."):
            retriever = SpreadsheetRetriever(
                knowledge_graph, 
                use_embeddings=True
            )
            
        # Step 3: Create query engine with the retriever
        with st.spinner("⚙️ Initializing query engine..."):
            engine = QueryEngine(retriever, spreadsheet_id=sheet_id)
            
        # Step 4: Process the query
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
- **Knowledge Graph**: Builds comprehensive metadata including cross-sheet references
- **Statistical Analysis**: Automatically calculates min/max/mean for numerical columns
- **Semantic Embeddings**: Uses sentence transformers for better semantic matching
- **Enhanced Context**: Provides Claude with rich statistical and structural information
""")