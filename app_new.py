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
        # Simple title with just the header and sheet info
        header = res.get('header', 'N/A')
        sheet_name = res.get('sheet', 'N/A')
        
        title = f"📌 Result {i}: {header} | {sheet_name}"
        
        with st.expander(title):
            st.markdown(f"**📋 Sheet:** `{sheet_name}`")
            st.markdown(f"**📊 Header:** `{header}`")
            st.markdown(f"**📍 Cell Range:** `{res.get('cell_range', 'N/A')}`")
            
            # Formula Information
            if res.get('formula'):
                normalized_formula = normalize_formula_display(res['formula'])
                st.markdown(f"**🔢 Formula:** {normalized_formula}", unsafe_allow_html=True)
            else:
                st.markdown("**🔢 Formula:** *No formula*")
            
            # Detailed Explanation
            if res.get('explanation'):
                st.markdown(f"**📝 Explanation:** {res['explanation']}")
            
            # Cross-sheet references (if available)
            if res.get('cross_sheet_refs') and res['cross_sheet_refs']:
                st.markdown(f"**🔄 Cross-sheet references:** {', '.join([f'`{ref}`' for ref in res['cross_sheet_refs']])}")

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
            
        # Build and cache the retriever with the knowledge graph
        with st.spinner("🧠 Building semantic retriever (includes embedding computation)..."):
            # Get cached sentence transformer model
            cached_model = get_cached_sentence_transformer()
            
            # Create retriever with the cached knowledge graph and model
            retriever = SpreadsheetRetriever(
                knowledge_graph, 
                use_embeddings=True,
                model=cached_model
            )
            
            # Cache the retriever
            st.session_state.retriever = retriever
            
        st.success(f"✅ Knowledge graph and retriever built for: {knowledge_graph.title}")
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

st.set_page_config(page_title="Semantic Spreadsheet Search", layout="wide")
st.title("🔍 Semantic Spreadsheet Search")
st.markdown("*Powered by enhanced knowledge graphs and statistical analysis*")

# Initialize session state
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'current_sheet_id' not in st.session_state:
    st.session_state.current_sheet_id = None
if 'current_sheet_name' not in st.session_state:
    st.session_state.current_sheet_name = None
if 'sentence_transformer_model' not in st.session_state:
    st.session_state.sentence_transformer_model = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'using_custom_sheet' not in st.session_state:
    st.session_state.using_custom_sheet = False
if 'custom_sheet_url' not in st.session_state:
    st.session_state.custom_sheet_url = ""

def get_cached_sentence_transformer():
    """Get or load the sentence transformer model from cache."""
    if st.session_state.sentence_transformer_model is None:
        try:
            with st.spinner("🤖 Loading sentence transformer model (one-time setup)..."):
                from sentence_transformers import SentenceTransformer
                st.session_state.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Sentence transformer model loaded and cached!")
        except ImportError:
            st.warning("⚠️ Sentence transformers not available. Using fallback similarity.")
            return None
        except Exception as e:
            st.error(f"❌ Error loading sentence transformer: {str(e)}")
            return None
    else:
        # Model is already cached
        st.info("🚀 Using cached sentence transformer model")
    return st.session_state.sentence_transformer_model

# Spreadsheet selector section
st.markdown("## 📊 Select Spreadsheet")

# Create three columns for different input methods with OR divider
col1, col_or, col2 = st.columns([5, 1, 5])

with col1:
    st.markdown("### 📋 Predefined Test Spreadsheets")
    spreadsheet_options = ["Please select a spreadsheet..."] + list(SPREADSHEETS.keys())
    sheet_choice = st.selectbox("Choose a spreadsheet:", spreadsheet_options, key="predefined_sheet")

with col_or:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some vertical spacing
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
            st.caption("⚠️ Clear to use predefined sheets")
            if clear_custom:
                st.session_state.using_custom_sheet = False
                st.session_state.custom_sheet_url = ""
                # Clear cached knowledge graph when clearing custom
                st.session_state.knowledge_graph = None
                st.session_state.current_sheet_id = None
                st.session_state.current_sheet_name = None
                st.session_state.retriever = None
                # Reset predefined tracking so dropdown can be used again
                st.session_state.previous_predefined_choice = "Please select a spreadsheet..."
                st.success("✅ Custom spreadsheet cleared!")

# Handle custom spreadsheet button click
if use_custom and custom_url.strip():
    st.session_state.using_custom_sheet = True
    st.session_state.custom_sheet_url = custom_url.strip()
    # Force rebuild by clearing cached knowledge graph when switching to custom
    st.session_state.knowledge_graph = None
    st.session_state.current_sheet_id = None
    st.session_state.current_sheet_name = None
    st.session_state.retriever = None
    st.success(f"✅ Custom spreadsheet selected!")
elif use_custom and not custom_url.strip():
    st.error("❌ Please provide a Google Sheets URL or ID for your custom spreadsheet.")

# Handle predefined spreadsheet selection (only when dropdown actually changes)
# We need to track the previous predefined selection to detect actual changes
if 'previous_predefined_choice' not in st.session_state:
    st.session_state.previous_predefined_choice = "Please select a spreadsheet..."

# Only process predefined selection if user actually changed the dropdown AND it's not a custom selection
if (sheet_choice != st.session_state.previous_predefined_choice and 
    sheet_choice != "Please select a spreadsheet..." and 
    not use_custom):  # Don't override if user just clicked custom button
    
    # Clear custom when switching to predefined
    if st.session_state.using_custom_sheet:
        # Force rebuild by clearing cached knowledge graph when switching to predefined
        st.session_state.knowledge_graph = None
        st.session_state.current_sheet_id = None
        st.session_state.current_sheet_name = None
        st.session_state.retriever = None
    st.session_state.using_custom_sheet = False
    st.session_state.custom_sheet_url = ""
    st.session_state.previous_predefined_choice = sheet_choice

# Determine which spreadsheet to use based on session state
sheet_id = None
sheet_name = None

if st.session_state.using_custom_sheet and st.session_state.custom_sheet_url:
    # User is using custom spreadsheet - this takes priority
    try:
        sheet_id = extract_sheet_id_from_url(st.session_state.custom_sheet_url)
        sheet_name = "Custom Spreadsheet"  # Will be updated with actual title when knowledge graph is built
        st.info(f"📋 Using custom spreadsheet with ID: {sheet_id}")
    except Exception as e:
        st.error(f"❌ Error processing custom spreadsheet URL/ID: {str(e)}")
        # Reset custom sheet state on error
        st.session_state.using_custom_sheet = False
        st.session_state.custom_sheet_url = ""
        st.session_state.previous_predefined_choice = "Please select a spreadsheet..."
        sheet_id = None
        sheet_name = None
elif not st.session_state.using_custom_sheet and sheet_choice != "Please select a spreadsheet...":
    # User selected predefined spreadsheet and not using custom
    sheet_id = SPREADSHEETS[sheet_choice]
    sheet_name = sheet_choice

# Only proceed if a valid spreadsheet is selected
if not sheet_id:
    knowledge_graph = None
    st.info("👆 Please select a predefined spreadsheet or add your own custom spreadsheet to begin")
else:
    # Check if user selected a different spreadsheet and build/cache knowledge graph
    if st.session_state.current_sheet_id != sheet_id:
        knowledge_graph = build_and_cache_knowledge_graph(sheet_id, sheet_name)
    else:
        knowledge_graph = st.session_state.knowledge_graph
        if knowledge_graph:
            st.info(f"📋 Using cached knowledge graph and retriever for: {knowledge_graph.title}")

# Show spreadsheet overview if knowledge graph is available
if knowledge_graph:
    with st.expander("📈 Spreadsheet Overview"):
        st.write(f"**Title:** {knowledge_graph.title}")
        st.write(f"**Sheets:** {len(knowledge_graph.sheets)}")
        for sheet_name_overview, sheet_meta in knowledge_graph.sheets.items():
            st.write(f"  - **{sheet_name_overview}**: {len(sheet_meta.columns)} columns")

# Only show search functionality if a valid spreadsheet is selected
if sheet_id:
    # Initialize query state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Query input
    st.markdown("## 🔍 Search Your Spreadsheet")
    
    # Sample questions
    st.markdown("### 💡 Try these sample questions:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💰 Find all revenue calculations", key="sample1"):
            st.session_state.current_query = "Find all revenue calculations"
        if st.button("💸 Show me cost-related formulas", key="sample2"):
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
    
    if st.button("Search") and query.strip():
        if not knowledge_graph:
            st.error("❌ Please select a spreadsheet first.")
            st.stop()
            
        try:
            # Use cached retriever (embeddings already computed)
            retriever = st.session_state.retriever
            if not retriever:
                st.error("❌ Retriever not available. Please try selecting the spreadsheet again.")
                st.stop()
                
            st.info("🚀 Using cached retriever with pre-computed embeddings")
                
            # Create query engine with the cached retriever
            with st.spinner("⚙️ Initializing query engine..."):
                engine = QueryEngine(retriever, spreadsheet_id=sheet_id)
                
            # Process the query
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