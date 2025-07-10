#!/usr/bin/env python3
"""
Example usage of the LangGraph-based search engine
"""

import os
from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced
from src.rag.retriever import SpreadsheetRetriever
from src.semantic_search.langgraph_search_engine import LangGraphSearchEngine


def main():
    """Demonstrate the LangGraph search engine"""
    
    # You'll need to set your Anthropic API key
    # export ANTHROPIC_API_KEY="your-key-here"
    
    print("🚀 Initializing LangGraph Search Engine...")
    
    # Setup - using the test spreadsheet
    TEST_SPREADSHEET_ID = "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc"
    
    # Parse spreadsheet and build knowledge graph
    parser = SpreadsheetParserAdvanced()
    spreadsheet = parser.parse(TEST_SPREADSHEET_ID)
    kg = parser.build_knowledge_graph(spreadsheet)
    
    # Create RAG retriever
    retriever = SpreadsheetRetriever(kg, debug=False)
    
    # Create LangGraph search engine
    search_engine = LangGraphSearchEngine(retriever)
    
    print("✅ Search engine ready!\n")
    
    # Example queries to test
    example_queries = [
        "show me revenue data",
        "what expenses do we have?", 
        "calculate total revenue",  # This will be classified as 'calculate'
        "find salary information",
        "what costs are tracked?"
    ]
    
    print("🧪 Testing with example queries:\n")
    
    for query in example_queries:
        print(f"{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        try:
            result = search_engine.search(query)
            print(f"Query Type: {result['query_type']}")
            print(f"Results Found: {result['num_results']}")
            print("\n" + result['response'])
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n")


if __name__ == "__main__":
    main() 