import streamlit as st
from src.data_ingestion.spreadsheet_parser import SpreadsheetParser
from src.semantic_search.search_engine import SemanticSearchEngine
import json

# Hardcoded spreadsheet titles and IDs
SPREADSHEETS = {
    "Sales Dashboard": "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls",
    "Financial Model": "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc"
}

st.set_page_config(page_title="Semantic Spreadsheet Search", layout="wide")
st.title("🔍 Semantic Spreadsheet Search")

# Spreadsheet selector
sheet_choice = st.selectbox("Choose a spreadsheet:", list(SPREADSHEETS.keys()))
sheet_id = SPREADSHEETS[sheet_choice]

query = st.text_input("Enter your semantic query:")

if st.button("Search") and query.strip():
    with st.spinner("🔍 Parsing spreadsheet..."):
        parser = SpreadsheetParser()
        parsed_data = parser.parse_spreadsheet(sheet_id)

    search_engine = SemanticSearchEngine()

    with st.spinner("🧠 Sending query to Claude..."):
        summaries = search_engine.extract_semantic_chunks(parsed_data)
        results_json = search_engine.search(query, summaries)

    st.success("🎯 Search Results")
    try:
        # Extract JSON from Claude's response
        json_start = results_json.find('[')
        json_end = results_json.rfind(']') + 1
        if json_start != -1 and json_end != 0:
            json_text = results_json[json_start:json_end]
            parsed_results = json.loads(json_text)
        else:
            # Fallback: try to parse the entire response
            parsed_results = json.loads(results_json)
        
        for res in parsed_results:
            with st.expander(f"📌 {res['concept_name']} in sheet '{res['sheet']}' → cell {res['cell']}"):
                st.markdown(f"**Header:** `{res['header']}`")
                st.markdown(f"**Formula:** `{res['formula']}`" if res['formula'] else "*No formula*")
                st.markdown(f"**Explanation:** {res['explanation']}")
    except Exception as e:
        st.error("❌ Failed to parse response from Claude.")
        st.code(results_json, language="json")
        st.exception(e)
