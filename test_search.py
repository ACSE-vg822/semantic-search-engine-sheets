# test_search.py

from src.data_ingestion.spreadsheet_parser import SpreadsheetParser
from src.semantic_search.search_engine import SemanticSearchEngine
import json

# Replace this with your test spreadsheet ID
SPREADSHEET_ID = "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"

def main():
    print("🔍 Parsing spreadsheet...")
    parser = SpreadsheetParser()
    spreadsheet_data = parser.parse_spreadsheet(SPREADSHEET_ID)

    print("✅ Spreadsheet parsed. Extracting semantic chunks...")
    search_engine = SemanticSearchEngine()
    chunks = search_engine.extract_semantic_chunks(spreadsheet_data)

    print(f"✅ Extracted {len(chunks)} chunks. Sample:")
    for chunk in chunks:
        print(json.dumps(chunk, indent=2))

    query = input("\n💬 Enter your semantic query: ")

    print("🧠 Sending query to Claude...")
    result = search_engine.search(query, chunks)

    print("\n🎯 Search Results:")
    print(result)

if __name__ == "__main__":
    main()
