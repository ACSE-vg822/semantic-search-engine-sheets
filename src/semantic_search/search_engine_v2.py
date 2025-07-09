# search_engine_v2.py - LangGraph-based Semantic Search with Pandas Analysis

import os
import json
import logging
import pandas as pd
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchState(TypedDict):
    """State schema for the LangGraph search workflow"""
    query: str
    retriever_results: List[Tuple]
    plan: Optional[Dict]
    enriched_data: List[Dict]
    insights: Dict
    status: str

@dataclass
class DataFetchPlan:
    """Plan for what data to fetch based on query analysis"""
    target_sheets: List[str]
    target_concepts: List[str] 
    analysis_type: str  # "summary", "trend", "comparison", "calculation"
    specific_rows: List[RowMetadata]
    specific_columns: List[ColumnMetadata]
    expected_insights: List[str]

@dataclass
class EnrichedRowData:
    """Row data with full context and resolved references"""
    first_cell_value: str
    sheet: str
    row_number: int
    values: List[Union[str, float, int]]
    formulas: List[str]
    headers: List[str]
    cell_addresses: str
    cross_references: Dict[str, any]
    metadata: Dict[str, any]

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
        create a focused plan for data fetching and analysis.

        Respond with a JSON object containing:
        {
            "target_sheets": ["sheet1", "sheet2"],
            "target_concepts": ["concept1", "concept2"], 
            "analysis_type": "summary|trend|comparison|calculation",
            "expected_insights": ["insight1", "insight2"]
        }
        
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
                expected_insights=plan_json.get("expected_insights", [])
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
                expected_insights=["totals", "key_metrics"]
            )
    
    def _summarize_available_data(self, rows: List[RowMetadata], columns: List[ColumnMetadata]) -> str:
        """Create summary of available data for Claude"""
        summary_parts = []
        
        if rows:
            summary_parts.append("Available Row Concepts:")
            for row in rows[:5]:  # Limit to prevent token overflow
                summary_parts.append(f"  - {row.first_cell_value} ({row.sheet}) - values: {row.sample_values[:3]}")
        
        if columns:
            summary_parts.append("\nAvailable Columns:")
            for col in columns[:5]:
                summary_parts.append(f"  - {col.header} ({col.sheet}) - values: {col.sample_values[:3]}")
                
        return "\n".join(summary_parts)

class DataFetcherNode:
    """Node 2: Fetch targeted data with full context"""
    
    def __init__(self, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id
        self.client = self._setup_google_sheets_auth()
    
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
            expected_insights=plan_dict["expected_insights"]
        )
        enriched_data = self.fetch_targeted_data(plan)
        state["enriched_data"] = [asdict(d) for d in enriched_data]
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
                enriched_row = self._fetch_live_column_data(col_meta, spreadsheet)
                enriched_data.append(enriched_row)
        
        # Only include rows if we don't have good column matches, or if rows are specifically relevant
        if not (matching_columns and query_is_business_metric):
            # No good columns found, fall back to rows with live data fetching
            rows_to_process = matching_rows if matching_rows else plan.specific_rows[:3]
            
            for row_meta in rows_to_process:
                enriched_row = self._fetch_live_row_data(row_meta, spreadsheet)
                enriched_data.append(enriched_row)
            
        return enriched_data
    
    def _fetch_live_column_data(self, col_meta: ColumnMetadata, spreadsheet) -> EnrichedRowData:
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
                    metadata={"data_type": col_meta.data_type, "source": "live_data", "column_header": col_meta.header}
                )
                
            except Exception as e:
                logger.error(f"Failed to fetch live column data for {col_meta.header}: {e}")
        
        # Fallback to cached metadata
        return self._create_enriched_from_column(col_meta)
    
    def _fetch_live_row_data(self, row_meta: RowMetadata, spreadsheet) -> EnrichedRowData:
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
                    metadata={"data_type": row_meta.data_type, "source": "live_data"}
                )
                
            except Exception as e:
                logger.error(f"Failed to fetch live row data for {row_meta.first_cell_value}: {e}")
        
        # Fallback to cached metadata
        return self._create_enriched_from_metadata(row_meta)
    
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
    


class InsightGeneratorNode:
    """Node 3: Generate insights using pandas analysis"""
    
    def __init__(self, claude_client):
        self.client = claude_client
    
    def __call__(self, state: SearchState) -> SearchState:
        """LangGraph node function for insight generation"""
        # Convert enriched_data dicts back to EnrichedRowData objects
        enriched_data = [EnrichedRowData(**d) for d in state["enriched_data"]]
        
        # Convert plan dict back to DataFetchPlan with proper object reconstruction
        plan_dict = state["plan"]
        plan = DataFetchPlan(
            target_sheets=plan_dict["target_sheets"],
            target_concepts=plan_dict["target_concepts"],
            analysis_type=plan_dict["analysis_type"],
            specific_rows=[RowMetadata(**row) for row in plan_dict["specific_rows"]],
            specific_columns=[ColumnMetadata(**col) for col in plan_dict["specific_columns"]],
            expected_insights=plan_dict["expected_insights"]
        )
        
        insights = self.generate_insights(enriched_data, plan, state["query"])
        state["insights"] = insights
        state["status"] = "success"
        return state
    
    def generate_insights(self, data: List[EnrichedRowData], plan: DataFetchPlan, original_query: str) -> Dict:
        """Generate business insights from the enriched data"""
        
        if not data:
            return {"error": "No data available for analysis"}
        
        # Convert to pandas DataFrame for analysis
        df = self._create_analysis_dataframe(data)
        
        insights = {
            "raw_data_summary": self._summarize_raw_data(data),
            "pandas_analysis": {}
        }
        
        # Perform analysis based on plan type
        if plan.analysis_type == "trend" and len(df.columns) > 2:
            insights["pandas_analysis"]["growth_analysis"] = self._calculate_growth_trends(df)
            
        elif plan.analysis_type == "summary":
            insights["pandas_analysis"]["key_metrics"] = self._summarize_key_metrics(df, data)
            
        elif plan.analysis_type == "calculation":
            insights["pandas_analysis"]["formula_breakdown"] = self._explain_calculations(data)
            
        elif plan.analysis_type == "comparison":
            insights["pandas_analysis"]["comparison_analysis"] = self._perform_comparison_analysis(df)
        
        # Generate business narrative using Claude
        insights["business_narrative"] = self._generate_business_narrative(insights, original_query)
        
        return insights
    
    def _create_analysis_dataframe(self, data: List[EnrichedRowData]) -> pd.DataFrame:
        """Convert enriched data to pandas DataFrame"""
        rows_dict = {}
        
        for row_data in data:
            # Use concept as row index
            concept = row_data.first_cell_value
            
            # Create columns from headers and values
            if len(row_data.headers) == len(row_data.values):
                row_dict = dict(zip(row_data.headers, row_data.values))
                rows_dict[concept] = row_dict
            else:
                # Fallback: use positional columns
                for i, value in enumerate(row_data.values):
                    if isinstance(value, (int, float)):  # Only include numeric values
                        rows_dict.setdefault(concept, {})[f"Value_{i+1}"] = value
        
        return pd.DataFrame.from_dict(rows_dict, orient='index')
    
    def _calculate_growth_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate growth trends using pandas"""
        try:
            import numpy as np
            numeric_df = df.select_dtypes(include=[np.number]) if hasattr(df, 'select_dtypes') else df._get_numeric_data()
            
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return {"error": "Insufficient numeric data for trend analysis"}
            
            # Calculate percentage changes
            pct_changes = numeric_df.pct_change(axis=1) * 100
            
            growth_analysis = {}
            for index in numeric_df.index:
                row_data = numeric_df.loc[index]
                growth_rates = pct_changes.loc[index].dropna()
                
                if not growth_rates.empty:
                    growth_analysis[index] = {
                        "values": row_data.tolist(),
                        "growth_rates": growth_rates.tolist(),
                        "average_growth": growth_rates.mean(),
                        "total_growth": ((row_data.iloc[-1] / row_data.iloc[0]) - 1) * 100 if row_data.iloc[0] != 0 else 0
                    }
            
            return growth_analysis
            
        except Exception as e:
            return {"error": f"Error in growth trend calculation: {str(e)}"}
    
    def _summarize_key_metrics(self, df: pd.DataFrame, data: List[EnrichedRowData]) -> Dict:
        """Summarize key metrics"""
        metrics = {}
        
        for row_data in data:
            concept = row_data.first_cell_value
            
            # Extract numeric values
            numeric_values = [v for v in row_data.values if isinstance(v, (int, float))]
            
            if numeric_values:
                metrics[concept] = {
                    "primary_value": numeric_values[0] if numeric_values else None,
                    "all_values": numeric_values,
                    "total": sum(numeric_values),
                    "average": sum(numeric_values) / len(numeric_values),
                    "sheet": row_data.sheet,
                    "has_formulas": len(row_data.formulas) > 0,
                    "cross_references": row_data.cross_references
                }
        
        return metrics
    
    def _explain_calculations(self, data: List[EnrichedRowData]) -> Dict:
        """Explain formulas and calculations"""
        explanations = {}
        
        for row_data in data:
            if row_data.formulas or row_data.cross_references:
                explanations[row_data.first_cell_value] = {
                    "formulas": row_data.formulas,
                    "cross_references": row_data.cross_references,
                    "explanation": f"This {row_data.first_cell_value} calculation involves {len(row_data.formulas)} formulas and {len(row_data.cross_references)} cross-references"
                }
                
        return explanations
    
    def _perform_comparison_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform comparison analysis"""
        try:
            numeric_df = df._get_numeric_data()
            
            if numeric_df.empty:
                return {"error": "No numeric data for comparison"}
            
            comparison = {
                "summary_stats": numeric_df.describe().to_dict(),
                "correlations": numeric_df.corr().to_dict() if len(numeric_df.columns) > 1 else {},
                "rankings": numeric_df.sum(axis=1).sort_values(ascending=False).to_dict()
            }
            
            return comparison
            
        except Exception as e:
            return {"error": f"Error in comparison analysis: {str(e)}"}
    
    def _summarize_raw_data(self, data: List[EnrichedRowData]) -> Dict:
        """Summarize the raw data structure"""
        return {
            "total_concepts": len(data),
            "sheets_involved": list(set([d.sheet for d in data])),
            "concepts_found": [d.first_cell_value for d in data],
            "has_formulas": sum(1 for d in data if d.formulas),
            "has_cross_references": sum(1 for d in data if d.cross_references)
        }
    
    def _generate_business_narrative(self, insights: Dict, original_query: str) -> str:
        """Generate business narrative using Claude"""
        try:
            system_prompt = """You are a business analyst. Given analysis results and the original user query, 
            provide a clear, business-focused narrative explanation. Focus on actionable insights."""
            
            user_prompt = f"""
            Original Query: "{original_query}"
            
            Analysis Results:
            {json.dumps(insights, indent=2, default=str)}
            
            Provide a clear business narrative explaining what was found and what it means.
            """
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Generated insights from {insights['raw_data_summary']['total_concepts']} concepts across {len(insights['raw_data_summary']['sheets_involved'])} sheets."

class SearchEngineV2:
    """LangGraph-based search engine"""
    
    def __init__(self, knowledge_graph: SpreadsheetKnowledgeGraph, spreadsheet_id: str):
        self.kg = knowledge_graph
        self.spreadsheet_id = spreadsheet_id
        
        # Initialize existing retriever for initial filtering
        self.retriever = SpreadsheetRetriever(knowledge_graph, use_embeddings=True)
        
        # Setup Claude client
        self.claude_client = self._setup_claude_client()
        
        # Initialize nodes
        self.query_analyzer = QueryAnalyzerNode(self.claude_client)
        self.data_fetcher = DataFetcherNode(spreadsheet_id)
        self.insight_generator = InsightGeneratorNode(self.claude_client)
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # Initialize the graph
        workflow = StateGraph(SearchState)
        
        # Add nodes
        workflow.add_node("query_analyzer", self.query_analyzer)
        workflow.add_node("data_fetcher", self.data_fetcher)
        workflow.add_node("insight_generator", self.insight_generator)
        
        # Define the flow
        workflow.set_entry_point("query_analyzer")
        workflow.add_edge("query_analyzer", "data_fetcher")
        workflow.add_edge("data_fetcher", "insight_generator")
        workflow.add_edge("insight_generator", END)
        
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
        
        # Initialize the state
        initial_state: SearchState = {
            "query": query,
            "retriever_results": retriever_results,
            "plan": None,
            "enriched_data": [],
            "insights": {},
            "status": "pending"
        }
        
        # Run the LangGraph workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "query": final_state["query"],
            "plan": final_state["plan"],
            "enriched_data": final_state["enriched_data"],
            "insights": final_state["insights"],
            "status": final_state["status"]
        } 