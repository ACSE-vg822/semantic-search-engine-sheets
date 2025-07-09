# types.py - Shared types for semantic search components

from typing import Dict, List, Optional, Union, Tuple, TypedDict
from dataclasses import dataclass

from src.data_ingestion.spreadsheet_parser_advance import RowMetadata, ColumnMetadata

class SearchState(TypedDict):
    """State schema for the LangGraph search workflow"""
    query: str
    retriever_results: List[Tuple]
    plan: Optional[Dict]
    enriched_data: List[Dict]
    calculation_result: Optional[Dict]
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
    # New: LLM-generated explanations for each data item
    data_explanations: Dict[str, str]  # key: "row_X" or "col_X", value: explanation
    # New: Branch decision for workflow routing
    branch_type: str  # "search" or "calculate"

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