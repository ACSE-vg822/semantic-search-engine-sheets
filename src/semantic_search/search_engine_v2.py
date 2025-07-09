# search_engine_v2.py - LangGraph-based Semantic Search

import os
import json
import logging

import toml
from typing import Dict, List, Optional, Union, Tuple, TypedDict
from dataclasses import dataclass, asdict

import gspread
import streamlit as st
from google.oauth2 import service_account
import anthropic
from langgraph.graph import StateGraph, END

from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetKnowledgeGraph, RowMetadata, ColumnMetadata
from src.rag.retriever import SpreadsheetRetriever
from src.semantic_search.types import SearchState, DataFetchPlan, EnrichedRowData
from src.semantic_search.calculator_node import CalculatorNode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryAnalyzerNode:
    """Node 1: Analyze query and create focused data fetching plan"""
    
    def __init__(self, claude_client):
        self.client = claude_client
    
    def __call__(self, state: SearchState) -> SearchState:
        """LangGraph node function for query analysis"""
        plan = self.analyze_query(state["query"], state["retriever_results"])
        state["plan"] = asdict(plan)
        return state
    
    def analyze_query(self, query: str, retriever_results: List[Tuple]) -> DataFetchPlan:
        """Analyze user query and RAG results to create focused fetching plan"""
        
        # Separate rows and columns from retriever results
        rows = [metadata for metadata, score, entry_type in retriever_results if entry_type == "row"]
        columns = [metadata for metadata, score, entry_type in retriever_results if entry_type == "column"]
        
        # Create summary of available data for Claude
        available_data = self._summarize_available_data(rows, columns)
        
        system_prompt = """You are a spreadsheet analysis planner. Given a user query and available data, 
        create a focused plan for data fetching and analysis. You must also determine the appropriate workflow branch.

        Respond with a JSON object containing:
        {
            "branch_type": "search|calculate",
            "target_sheets": ["sheet1", "sheet2"],
            "target_concepts": ["concept1", "concept2"], 
            "analysis_type": "summary|trend|comparison|calculation",
            "expected_insights": ["insight1", "insight2"],
            "data_explanations": {
                "row_Sheet1_Revenue": "explanation for why this row is relevant",
                "col_Sheet1_Actual_Revenue": "explanation for why this column is relevant"
            }
        }
        
        BRANCH DECISION RULES:
        - "search": Use when user wants to FIND existing data, patterns, or information
          Examples: "find all percentage calculations", "show me revenue rows", "what expenses do we have"
        - "calculate": Use when user wants to COMPUTE new values or perform mathematical operations
          Examples: "calculate max revenue", "sum all costs", "what's the average profit margin"
        
        For data_explanations, provide a natural language explanation for each relevant row/column 
        explaining WHY it matches the user's query and what insights it can provide.
        Use the exact keys shown in the available data (like "row_Sheet1_Revenue", "col_Sheet1_Actual_Revenue").
        
        Analysis types:
        - summary: Basic totals and key metrics
        - trend: Growth rates, projections, time series
        - comparison: Variance analysis, benchmarking
        - calculation: Formula breakdown, derived metrics
        """
        
        user_prompt = f"""
        User Query: "{query}"
        
        Available Data:
        {available_data}
        
        Create a focused analysis plan. What specific concepts should we fetch and analyze?
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Using faster model for planning
                max_tokens=1000,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            plan_json = json.loads(response.content[0].text)
            
            return DataFetchPlan(
                target_sheets=plan_json.get("target_sheets", []),
                target_concepts=plan_json.get("target_concepts", []),
                analysis_type=plan_json.get("analysis_type", "summary"),
                specific_rows=rows,  # Pass through the filtered rows
                specific_columns=columns,  # Pass through the filtered columns
                expected_insights=plan_json.get("expected_insights", []),
                data_explanations=plan_json.get("data_explanations", {}),  # Extract LLM explanations
                branch_type=plan_json.get("branch_type", "search")  # Default to search if not specified
            )
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Fallback plan
            return DataFetchPlan(
                target_sheets=list(set([r.sheet for r in rows])),
                target_concepts=[r.first_cell_value for r in rows[:3]],
                analysis_type="summary",
                specific_rows=rows,
                specific_columns=columns,
                expected_insights=["totals", "key_metrics"],
                data_explanations={},  # Initialize with empty dict
                branch_type="search"  # Default fallback to search
            )
    
    def _summarize_available_data(self, rows: List[RowMetadata], columns: List[ColumnMetadata]) -> str:
        """Create indexed summary of available data for Claude"""
        summary_parts = []
        
        if rows:
            summary_parts.append("Available Row Concepts:")
            for row in rows[:5]:  # Limit to prevent token overflow
                row_key = f"row_{row.sheet}_{row.first_cell_value}".replace(" ", "_")
                summary_parts.append(f"  {row_key}: {row.first_cell_value} ({row.sheet}) - values: {row.sample_values[:3]}")
        
        if columns:
            summary_parts.append("\nAvailable Columns:")
            for col in columns[:5]:
                col_key = f"col_{col.sheet}_{col.header}".replace(" ", "_")
                summary_parts.append(f"  {col_key}: {col.header} ({col.sheet}) - values: {col.sample_values[:3]}")
                
        return "\n".join(summary_parts)

class DataFetcherNode:
    """Node 2: Fetch targeted data with full context"""
    
    def __init__(self, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id
        self.client = self._setup_google_sheets_auth()
        self.claude_client = None  # Will be set by SearchEngineV2
    
    def __call__(self, state: SearchState) -> SearchState:
        """LangGraph node function for data fetching"""
        plan_dict = state["plan"]
        # Convert dict back to DataFetchPlan with proper object reconstruction
        plan = DataFetchPlan(
            target_sheets=plan_dict["target_sheets"],
            target_concepts=plan_dict["target_concepts"],
            analysis_type=plan_dict["analysis_type"],
            specific_rows=[RowMetadata(**row) for row in plan_dict["specific_rows"]],
            specific_columns=[ColumnMetadata(**col) for col in plan_dict["specific_columns"]],
            expected_insights=plan_dict["expected_insights"],
            data_explanations=plan_dict["data_explanations"],  # Pass through explanations
            branch_type=plan_dict.get("branch_type", "search")  # Include branch type
        )
        enriched_data = self.fetch_targeted_data(plan)
        state["enriched_data"] = [asdict(d) for d in enriched_data]
        state["status"] = "success"  # Set success status since this is the final node
        return state
    
    def _setup_google_sheets_auth(self):
        """Setup Google Sheets authentication"""
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        return gspread.authorize(credentials)
    
    def fetch_targeted_data(self, plan: DataFetchPlan) -> List[EnrichedRowData]:
        """Fetch specific data based on the plan"""
        enriched_data = []
        
        # Get the live spreadsheet connection
        try:
            spreadsheet = self.client.open_by_key(self.spreadsheet_id)
        except Exception as e:
            logger.error(f"Failed to connect to spreadsheet {self.spreadsheet_id}: {e}")
            spreadsheet = None
        
        # Check for exact matches in both rows and columns
        matching_rows = []
        matching_columns = []
        
        for row_meta in plan.specific_rows:
            if any(keyword.lower() in row_meta.first_cell_value.lower() for keyword in plan.target_concepts):
                matching_rows.append(row_meta)
        
        for col_meta in plan.specific_columns:
            if any(keyword.lower() in col_meta.header.lower() for keyword in plan.target_concepts):
                matching_columns.append(col_meta)
        
        # Prioritize columns for typical business metrics (revenue, profit, cost, etc.)
        business_metrics = ['revenue', 'profit', 'cost', 'expense', 'sales', 'income', 'margin', 'growth']
        query_is_business_metric = any(metric in ' '.join(plan.target_concepts).lower() for metric in business_metrics)
        
        if matching_columns and query_is_business_metric:
            # Process matching columns with live data fetching
            for col_meta in matching_columns:
                col_key = f"col_{col_meta.sheet}_{col_meta.header}".replace(" ", "_")
                explanation = plan.data_explanations.get(col_key, "Relevant column data")
                enriched_row = self._fetch_live_column_data(col_meta, spreadsheet, explanation)
                enriched_data.append(enriched_row)
        
        # Only include rows if we don't have good column matches, or if rows are specifically relevant
        if not (matching_columns and query_is_business_metric):
            # No good columns found, fall back to rows with live data fetching
            rows_to_process = matching_rows if matching_rows else plan.specific_rows[:3]
            
            for row_meta in rows_to_process:
                row_key = f"row_{row_meta.sheet}_{row_meta.first_cell_value}".replace(" ", "_")
                explanation = plan.data_explanations.get(row_key, "Relevant row data")
                enriched_row = self._fetch_live_row_data(row_meta, spreadsheet, explanation)
                enriched_data.append(enriched_row)
            
        return enriched_data
    
    def _fetch_live_column_data(self, col_meta: ColumnMetadata, spreadsheet, explanation: str) -> EnrichedRowData:
        """Fetch live column data from spreadsheet with fallback to cached"""
        if spreadsheet:
            try:
                worksheet = spreadsheet.worksheet(col_meta.sheet)
                
                # Get live values from the column range
                live_values = worksheet.get(col_meta.addresses, value_render_option='UNFORMATTED_VALUE')
                live_formulas = worksheet.get(col_meta.addresses, value_render_option='FORMULA')
                
                # Flatten the values and formulas
                flat_values = [cell for row in live_values for cell in row if cell not in [None, ""]]
                flat_formulas = [cell for row in live_formulas for cell in row if cell and str(cell).startswith("=")]
                
                # Resolve cross-references if any formulas exist
                cross_refs = {}
                if flat_formulas:
                    cross_refs = self._resolve_formula_references(flat_formulas, spreadsheet)
                
                return EnrichedRowData(
                    first_cell_value=f"Column: {col_meta.header}",
                    sheet=col_meta.sheet,
                    row_number=0,  # Not applicable for columns
                    values=flat_values,
                    formulas=flat_formulas,
                    headers=[col_meta.header],
                    cell_addresses=col_meta.addresses,
                    cross_references=cross_refs,
                    metadata={
                        "data_type": col_meta.data_type, 
                        "source": "live_data", 
                        "column_header": col_meta.header,
                        "explanation": explanation
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to fetch live column data for {col_meta.header}: {e}")
        
        # Fallback to cached metadata
        fallback_data = self._create_enriched_from_column(col_meta)
        # Add LLM explanation to fallback as well
        fallback_data.metadata["explanation"] = explanation
        return fallback_data
    
    def _fetch_live_row_data(self, row_meta: RowMetadata, spreadsheet, explanation: str) -> EnrichedRowData:
        """Fetch live row data from spreadsheet with fallback to cached"""
        if spreadsheet:
            try:
                worksheet = spreadsheet.worksheet(row_meta.sheet)
                
                # Get live values from the row range
                live_values = worksheet.get(row_meta.cell_addresses, value_render_option='UNFORMATTED_VALUE')
                live_formulas = worksheet.get(row_meta.cell_addresses, value_render_option='FORMULA')
                
                # Flatten the values and formulas
                flat_values = [cell for row in live_values for cell in row if cell not in [None, ""]]
                flat_formulas = [cell for row in live_formulas for cell in row if cell and str(cell).startswith("=")]
                
                # Resolve cross-references if any formulas exist
                cross_refs = {}
                if flat_formulas:
                    cross_refs = self._resolve_formula_references(flat_formulas, spreadsheet)
                
                return EnrichedRowData(
                    first_cell_value=row_meta.first_cell_value,
                    sheet=row_meta.sheet,
                    row_number=row_meta.row_number,
                    values=flat_values,
                    formulas=flat_formulas,
                    headers=row_meta.col_headers,
                    cell_addresses=row_meta.cell_addresses,
                    cross_references=cross_refs,
                    metadata={
                        "data_type": row_meta.data_type, 
                        "source": "live_data",
                        "explanation": explanation
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to fetch live row data for {row_meta.first_cell_value}: {e}")
        
        # Fallback to cached metadata
        fallback_data = self._create_enriched_from_metadata(row_meta)
        # Add LLM explanation to fallback as well
        fallback_data.metadata["explanation"] = explanation
        return fallback_data
    

    
    def _resolve_formula_references(self, formulas: List[str], spreadsheet) -> Dict[str, any]:
        """Resolve formula references to actual values"""
        cross_refs = {}
        
        for formula in formulas:
            if "!" in formula:  # Cross-sheet reference
                try:
                    # Extract sheet and cell reference
                    parts = formula.replace("=", "").replace("'", "").split("!")
                    if len(parts) == 2:
                        sheet_name, cell_ref = parts
                        worksheet = spreadsheet.worksheet(sheet_name)
                        value = worksheet.get(cell_ref, value_render_option='UNFORMATTED_VALUE')
                        if value:
                            cross_refs[f"{sheet_name}!{cell_ref}"] = value[0][0] if value[0] else None
                except Exception as e:
                    pass
                    
        return cross_refs
    
    def _create_enriched_from_column(self, col_meta: ColumnMetadata) -> EnrichedRowData:
        """Create enriched row data from column metadata for column-major analysis"""
        return EnrichedRowData(
            first_cell_value=f"Column: {col_meta.header}",  # Make it clear this is a column
            sheet=col_meta.sheet,
            row_number=0,  # Not applicable for columns
            values=col_meta.sample_values,
            formulas=[col_meta.first_cell_formula] if col_meta.first_cell_formula else [],
            headers=[col_meta.header],
            cell_addresses=col_meta.addresses,  # This should be like "C2:C13" 
            cross_references=dict(zip([col_meta.first_cell_formula] if col_meta.first_cell_formula else [], ["column_formula"])),
            metadata={"data_type": col_meta.data_type, "source": "column_metadata", "column_header": col_meta.header}
        )
    
    def _create_enriched_from_metadata(self, row_meta: RowMetadata) -> EnrichedRowData:
        """Create enriched row data from metadata when live fetching fails"""
        return EnrichedRowData(
            first_cell_value=row_meta.first_cell_value,
            sheet=row_meta.sheet,
            row_number=row_meta.row_number,
            values=row_meta.sample_values,
            formulas=row_meta.formulae,
            headers=row_meta.col_headers,
            cell_addresses=row_meta.cell_addresses,  # This might be a string range
            cross_references=dict(zip(row_meta.formulae, ["cached_value"] * len(row_meta.formulae))),
            metadata={"data_type": row_meta.data_type, "source": "cached_metadata"}
        )

class SearchEngineV2:
    """LangGraph-based search engine"""
    
    def __init__(self, knowledge_graph: SpreadsheetKnowledgeGraph, spreadsheet_id: str):
        self.kg = knowledge_graph
        self.spreadsheet_id = spreadsheet_id
        
        # Initialize existing retriever for initial filtering
        self.retriever = SpreadsheetRetriever(knowledge_graph, use_embeddings=True)
        
        # Setup Claude client
        self.claude_client = self._setup_claude_client()
        
        # Initialize nodes (now 3 nodes: analyzer, fetcher, calculator)
        self.query_analyzer = QueryAnalyzerNode(self.claude_client)
        self.data_fetcher = DataFetcherNode(spreadsheet_id)
        self.data_fetcher.claude_client = self.claude_client  # Pass Claude client for explanations
        self.calculator = CalculatorNode(self.claude_client)
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with conditional branching"""
        # Initialize the graph
        workflow = StateGraph(SearchState)
        
        # Add nodes (3 nodes: analyzer, fetcher, calculator)
        workflow.add_node("query_analyzer", self.query_analyzer)
        workflow.add_node("data_fetcher", self.data_fetcher)
        workflow.add_node("calculator", self.calculator)
        
        # Define the conditional branching logic
        def route_after_analysis(state: SearchState) -> str:
            """Decide which branch to take based on analysis"""
            plan = state.get("plan", {})
            branch_type = plan.get("branch_type", "search")
            
            if branch_type == "calculate":
                return "data_fetcher"  # Go to data fetcher first for calculations
            else:
                return "search_complete"  # Skip data fetching for search queries
        
        def route_after_fetching(state: SearchState) -> str:
            """Route to calculator after data fetching"""
            return "calculator"
        
        # Define the flow with conditional branching
        workflow.set_entry_point("query_analyzer")
        workflow.add_conditional_edges(
            "query_analyzer",
            route_after_analysis,
            {
                "data_fetcher": "data_fetcher",
                "search_complete": END
            }
        )
        workflow.add_conditional_edges(
            "data_fetcher", 
            route_after_fetching,
            {
                "calculator": "calculator"
            }
        )
        workflow.add_edge("calculator", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def _setup_claude_client(self):
        """Setup Claude API client"""
        api_key = self._load_claude_key()
        if not api_key:
            raise ValueError("Claude API key not found")
            
        return anthropic.Anthropic(api_key=api_key)
    
    def _load_claude_key(self) -> str:
        """Load Claude API key from secrets"""
        try:
            return st.secrets["claude_api_key"]
        except:
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                secrets_path = os.path.join(project_root, '.streamlit', 'secrets.toml')
                if os.path.exists(secrets_path):
                    secrets = toml.load(secrets_path)
                    return secrets.get("claude_api_key", "")
            except:
                pass
        return ""
    
    def search(self, query: str) -> Dict:
        """Main search method using LangGraph pipeline"""
        
        # Use existing RAG retriever for initial filtering
        retriever_results = self.retriever.retrieve(query, top_k=10)
        
        # Initialize the state (includes calculation support)
        initial_state: SearchState = {
            "query": query,
            "retriever_results": retriever_results,
            "plan": None,
            "enriched_data": [],
            "calculation_result": None,
            "status": "pending"
        }
        
        # Run the LangGraph workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Set status to success since we completed data fetching
        final_state["status"] = "success"
        
        return {
            "query": final_state["query"],
            "plan": final_state["plan"],
            "enriched_data": final_state["enriched_data"],
            "calculation_result": final_state.get("calculation_result"),
            "status": final_state["status"]
        } 