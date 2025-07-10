import json
import logging
from typing import Dict, List, Tuple, Union, Literal, Optional
from dataclasses import dataclass, asdict
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

from src.rag.retriever import SpreadsheetRetriever
from src.data_ingestion.spreadsheet_parser_advance import ColumnMetadata, RowMetadata
import streamlit as st


# State structure for the graph
class SearchState(TypedDict):
    user_query: str
    query_type: Optional[Literal["search", "calculate"]]
    rag_results: Optional[List[Tuple[Union[ColumnMetadata, RowMetadata], float, str]]]
    filtered_results: Optional[List[Dict]]
    explanation: Optional[str]
    final_response: Optional[str]


@dataclass
class LangGraphSearchEngine:
    """Simple LangGraph-based search engine for spreadsheet data"""
    
    def __init__(self, retriever: SpreadsheetRetriever, llm_model: str = "claude-3-haiku-20240307"):
        self.retriever = retriever
        self.api_key = st.secrets["claude_api_key"]
        self.llm = ChatAnthropic(model=llm_model, temperature=0, api_key=self.api_key)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(SearchState)
        
        # Add nodes
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("search", self._search_node)
        
        # Add a placeholder calculate node
        workflow.add_node("calculate_placeholder", self._calculate_placeholder_node)
        
        # Add edges
        workflow.set_entry_point("classifier")
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "search": "search",
                "calculate": "calculate_placeholder",
            }
        )
        workflow.add_edge("search", END)
        workflow.add_edge("calculate_placeholder", END)
        
        return workflow.compile()
    
    def _classifier_node(self, state: SearchState) -> SearchState:
        """First node: Classify query as 'search' or 'calculate'"""
        
        system_prompt = """You are a query classifier for a spreadsheet search engine.
        
Your job is to classify user queries into exactly one of these categories:
- "search": User wants to find or explore data (e.g., "show me revenue data", "what expenses do we have")
- "calculate": User wants to perform calculations (e.g., "calculate total revenue", "what's the profit margin")

Respond with ONLY the word "search" or "calculate" - nothing else."""

        user_message = f"Classify this query: {state['user_query']}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        query_type = response.content.strip().lower()
        
        # Ensure valid classification
        if query_type not in ["search", "calculate"]:
            query_type = "search"  # Default fallback
        
        print(f"🔍 Query classified as: {query_type}")
        
        return {
            **state,
            "query_type": query_type
        }
    
    def _search_node(self, state: SearchState) -> SearchState:
        """Search node: Retrieve data and filter with LLM explanation"""
        
        # Get RAG results
        rag_results = self.retriever.retrieve(state["user_query"], top_k=10)
        
        # Format results for LLM analysis
        formatted_results = []
        for i, (meta, score, result_type) in enumerate(rag_results):
            if result_type == "column":
                result_info = {
                    "index": i,
                    "type": "column",
                    "name": meta.header,
                    "sheet": meta.sheet,
                    "data_type": meta.data_type,
                    "sample_values": meta.sample_values,
                    "addresses": meta.addresses,
                    "score": score
                }
            else:  # row
                result_info = {
                    "index": i,
                    "type": "row", 
                    "name": meta.first_cell_value,
                    "sheet": meta.sheet,
                    "data_type": meta.data_type,
                    "sample_values": meta.sample_values,
                    "addresses": meta.cell_addresses,
                    "score": score
                }
            formatted_results.append(result_info)
        
        # LLM analysis prompt
        system_prompt = """You are an expert data analyst helping users understand spreadsheet search results.

Your task is to:
1. Analyze which results are truly relevant to the user's query
2. Explain WHY each relevant result matters
3. Filter out any irrelevant results
4. Provide a clear explanation

CRITICAL: You MUST respond with ONLY valid JSON. No other text before or after.

Return your response in this exact JSON format:
{
    "relevant_results": [
        {
            "index": 0,
            "relevance_explanation": "This result is relevant because..."
        },
        {
            "index": 2,
            "relevance_explanation": "This result matches the query because..."
        }
    ],
    "summary": "Found 2 relevant results related to the user's query about..."
}

Example response:
{
    "relevant_results": [
        {
            "index": 0,
            "relevance_explanation": "Revenue column contains the financial data the user is looking for"
        }
    ],
    "summary": "Found 1 revenue-related result that matches the query"
}

Remember: 
- Only include truly relevant results
- Use the exact index numbers from the input
- Provide clear explanations
- Return ONLY JSON, nothing else"""

        user_message = f"""
User Query: "{state['user_query']}"

Search Results:
{json.dumps(formatted_results, indent=2)}

Please analyze these results and return only the relevant ones with explanations.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Clean the response in case there's extra whitespace or formatting
            clean_response = response.content.strip()
            
            # Try to extract JSON if it's wrapped in markdown code blocks
            if clean_response.startswith("```json"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response.replace("```", "").strip()
            
            llm_analysis = json.loads(clean_response)
            
            # Validate the structure
            if "relevant_results" not in llm_analysis or "summary" not in llm_analysis:
                raise ValueError("Missing required fields in LLM response")
            
            # Filter original results based on LLM analysis
            filtered_results = []
            for result_info in llm_analysis["relevant_results"]:
                original_index = result_info["index"]
                if 0 <= original_index < len(formatted_results):
                    original_result = formatted_results[original_index].copy()
                    original_result["relevance_explanation"] = result_info["relevance_explanation"]
                    filtered_results.append(original_result)
                else:
                    print(f"⚠️ Invalid index {original_index} from LLM, skipping")
            
            explanation = llm_analysis.get("summary", "Results filtered and analyzed.")
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback if LLM doesn't return valid JSON
            print(f"⚠️ LLM response parsing failed: {e}")
            print(f"Raw LLM response: {response.content[:200]}...")
            filtered_results = formatted_results[:5]  # Just take top 5 as fallback
            explanation = f"LLM filtering failed ({str(e)}), showing top 5 results"
        
        print(f"📊 Found {len(filtered_results)} relevant results out of {len(formatted_results)} total")
        
        return {
            **state,
            "rag_results": rag_results,
            "filtered_results": filtered_results,
            "explanation": explanation,
            "final_response": self._format_final_response(filtered_results, explanation)
        }
    
    def _calculate_placeholder_node(self, state: SearchState) -> SearchState:
        """Placeholder node for calculate queries - to be implemented later"""
        
        print("🧮 Calculate functionality not yet implemented")
        
        final_response = f"""🧮 **Calculate Query Detected**

Query: "{state['user_query']}"

⚠️ **Feature Coming Soon**: The calculate functionality is not yet implemented. 

For now, this query has been identified as a calculation request. In the future, this will:
- Parse mathematical expressions from your query
- Identify relevant data from the spreadsheet
- Perform the requested calculations
- Return computed results

Try rephrasing as a search query instead, like:
- "show me revenue data" 
- "find financial information"
- "what revenue numbers do we have"
"""
        
        return {
            **state,
            "rag_results": None,
            "filtered_results": [],
            "explanation": "Calculate functionality placeholder",
            "final_response": final_response
        }
    
    def _route_query(self, state: SearchState) -> str:
        """Router function to determine next node based on query type"""
        return state["query_type"]
    
    def _format_final_response(self, filtered_results: List[Dict], explanation: str) -> str:
        """Format the final response for the user"""
        if not filtered_results:
            return "❌ No relevant results found for your query."
        
        response = f"🔍 **Search Results Analysis:**\n\n{explanation}\n\n"
        response += f"**Found {len(filtered_results)} relevant result(s):**\n\n"
        
        for i, result in enumerate(filtered_results, 1):
            response += f"**{i}. {result['name']}** ({result['type']})\n"
            response += f"   📍 Sheet: {result['sheet']}\n"
            #response += f"   📊 Type: {result['data_type']}\n"
            response += f"   📍 Range: {result.get('addresses', 'N/A')}\n"
            response += f"   🎯 Relevance: {result.get('relevance_explanation', 'N/A')}\n"
            response += f"   📝 Sample: {', '.join(map(str, result['sample_values'][:3]))}\n\n"
        
        return response
    
    def search(self, query: str) -> Dict:
        """Main search interface"""
        initial_state = SearchState(
            user_query=query,
            query_type=None,
            rag_results=None,
            filtered_results=None,
            explanation=None,
            final_response=None
        )
        
        print(f"🚀 Processing query: '{query}'")
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "query_type": final_state["query_type"],
            "num_results": len(final_state["filtered_results"]) if final_state["filtered_results"] else 0,
            "explanation": final_state["explanation"],
            "response": final_state["final_response"],
            "raw_results": final_state["filtered_results"]
        }


# 🧪 Test block
if __name__ == "__main__":
    from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced
    
    # Initialize components
    TEST_SPREADSHEET_ID = "1EvWvbiJIIIASse3b9iHP1JAOTmnw3Xur7oRpG-o9Oxc"
    
    print("🔧 Setting up search engine...")
    parser = SpreadsheetParserAdvanced()
    spreadsheet = parser.parse(TEST_SPREADSHEET_ID)
    kg = parser.build_knowledge_graph(spreadsheet)
    retriever = SpreadsheetRetriever(kg, debug=True)
    
    # Create LangGraph search engine
    search_engine = LangGraphSearchEngine(retriever)
    
    print("✅ Search engine ready!")
    
    # Interactive testing
    while True:
        query = input("\n🔍 Enter your query (or 'quit' to exit): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if query:
            try:
                result = search_engine.search(query)
                print("\n" + "="*50)
                print(result["response"])
                print("="*50)
            except Exception as e:
                print(f"❌ Error: {e}") 