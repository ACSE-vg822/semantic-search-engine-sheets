import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
import streamlit as st

from src.rag.retriever import SpreadsheetRetriever
from src.data_ingestion.spreadsheet_parser_advance import ColumnMetadata, RowMetadata, SpreadsheetData, SpreadsheetParserAdvanced


@dataclass
class CalculationRequest:
    """Parsed calculation request from natural language query"""
    operation: Literal["sum", "average", "count", "min", "max", "divide", "multiply", "subtract", "add", "percentage"]
    target_concepts: List[str]  # e.g., ["revenue", "sales"] 
    filters: Optional[Dict[str, str]] = None  # e.g., {"period": "Q1", "department": "sales"}
    original_query: str = ""


@dataclass 
class CalculationResult:
    """Result of a calculation operation"""
    value: float
    operation: str
    data_used: List[Dict]  # Details about data sources used
    explanation: str
    formatted_result: str


class CalculationEngine:
    """Engine for parsing and executing calculations on spreadsheet data"""
    
    def __init__(self, retriever: SpreadsheetRetriever, spreadsheet_id: str, llm_model: str = "claude-3-haiku-20240307"):
        self.retriever = retriever
        self.spreadsheet_id = spreadsheet_id
        self.api_key = st.secrets["claude_api_key"]
        self.llm = ChatAnthropic(model=llm_model, temperature=0, api_key=self.api_key)
        self._parser = None  # Lazy initialization
    
    def _get_parser(self) -> SpreadsheetParserAdvanced:
        """Get or create the spreadsheet parser (lazy initialization)"""
        if self._parser is None:
            self._parser = SpreadsheetParserAdvanced()
        return self._parser
    
    def _get_fresh_spreadsheet_data(self) -> SpreadsheetData:
        """Fetch fresh spreadsheet data from Google Sheets"""
        parser = self._get_parser()
        return parser.parse(self.spreadsheet_id)
    
    def parse_calculation_query(self, query: str) -> CalculationRequest:
        """Parse natural language query into structured calculation request"""
        
        system_prompt = """You are a calculation query parser for spreadsheet data.

Your job is to parse natural language calculation queries into structured requests.

Identify:
1. OPERATION: The mathematical operation (sum, average, count, min, max, divide, multiply, subtract, add, percentage)
2. TARGET_CONCEPTS: The business concepts to calculate with their full modifiers (e.g., "actual revenue", "target revenue", "net profit", "gross sales")
3. FILTERS: Any conditions or filters mentioned (e.g., time periods, categories, sheet names)

IMPORTANT RULES:
- Include ALL modifying words with concepts (e.g., "actual revenue" not just "revenue", "net profit" not just "profit")
- If a specific sheet is mentioned (e.g., "in Sheet <sheet name>"), include it in filters as {"sheet": "sheet_name"}
- Preserve the full specificity of business terms

Return ONLY valid JSON in this exact format:
{
    "operation": "sum|average|count|min|max|divide|multiply|subtract|add|percentage",
    "target_concepts": ["concept1", "concept2"],
    "filters": {"filter_key": "filter_value"} or null,
    "original_query": "the original query text"
}

Examples:
- "calculate total revenue" → {"operation": "sum", "target_concepts": ["revenue"], "filters": null}
- "calculate maximum actual revenue" → {"operation": "max", "target_concepts": ["actual revenue"], "filters": null}
- "calculate total revenue in Sheet: <sheet name>" → {"operation": "sum", "target_concepts": ["revenue"], "filters": {"sheet": "3-Year Forecast"}}
- "what's the average net profit amount?" → {"operation": "average", "target_concepts": ["net profit"], "filters": null}
- "sum all Q1 target sales" → {"operation": "sum", "target_concepts": ["target sales"], "filters": {"period": "Q1"}}
- "gross profit margin percentage" → {"operation": "percentage", "target_concepts": ["gross profit", "revenue"], "filters": null}

Respond with ONLY JSON, no other text."""
        
        user_message = f"Parse this calculation query: {query}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Clean response and parse JSON
            clean_response = response.content.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response.replace("```", "").strip()
            
            parsed = json.loads(clean_response)
            
            return CalculationRequest(
                operation=parsed["operation"],
                target_concepts=parsed["target_concepts"],
                filters=parsed.get("filters"),
                original_query=parsed["original_query"]
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: simple keyword matching
            return self._fallback_parse(query)
    
    def _fallback_parse(self, query: str) -> CalculationRequest:
        """Fallback parser using simple keyword matching"""
        query_lower = query.lower()
        
        # Detect operation
        if any(word in query_lower for word in ["sum", "total", "add up"]):
            operation = "sum"
        elif any(word in query_lower for word in ["average", "avg", "mean"]):
            operation = "average"  
        elif any(word in query_lower for word in ["count", "number of", "how many"]):
            operation = "count"
        elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
            operation = "min"
        elif any(word in query_lower for word in ["maximum", "max", "highest"]):
            operation = "max"
        elif any(word in query_lower for word in ["percentage", "percent", "ratio", "margin"]):
            operation = "percentage"
        else:
            operation = "sum"  # Default
        
        # Extract potential concepts (simple keyword extraction)
        concepts = []
        concept_keywords = ["revenue", "sales", "profit", "expense", "cost", "income", "margin", "growth"]
        for keyword in concept_keywords:
            if keyword in query_lower:
                concepts.append(keyword)
        
        if not concepts:
            concepts = ["total"]  # Default
        
        # Extract sheet name if mentioned
        filters = None
        import re
        sheet_match = re.search(r"(?:in\s+sheet:?\s*|sheet:?\s*)(['\"]?)([^'\"]+)\1", query_lower)
        if sheet_match:
            sheet_name = sheet_match.group(2).strip()
            filters = {"sheet": sheet_name}
        
        return CalculationRequest(
            operation=operation,
            target_concepts=concepts,
            filters=filters,
            original_query=query
        )
    
    def identify_relevant_data(self, calc_request: CalculationRequest) -> List[Tuple[Union[ColumnMetadata, RowMetadata], float, str]]:
        """Use RAG retriever to find relevant data for calculation"""
        
        # Build search query from calculation request
        search_queries = []
        for concept in calc_request.target_concepts:
            search_queries.append(f"find {concept} data")
            search_queries.append(f"{concept} calculations")
            search_queries.append(f"{concept} numbers")
        
        # Get all relevant results
        all_results = []
        for search_query in search_queries:
            results = self.retriever.retrieve(search_query, top_k=10)
            all_results.extend(results)
        
        # Remove duplicates and sort by relevance
        seen = set()
        unique_results = []
        for meta, score, result_type in all_results:
            # Create a unique identifier for deduplication
            if result_type == "column":
                identifier = f"{meta.sheet}_{meta.header}"
            else:
                identifier = f"{meta.sheet}_{meta.first_cell_value}_{meta.row_number}"
            
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append((meta, score, result_type))
        
        # Apply filters
        filtered_results = []
        
        for meta, score, result_type in unique_results:
            
            # Filter by sheet if specified
            if calc_request.filters and "sheet" in calc_request.filters:
                target_sheet = calc_request.filters["sheet"]
                if meta.sheet != target_sheet:
                    continue
            
            # Filter by concept precision - avoid derivative concepts
            # For each target concept, check if this is a precise match
            is_precise_match = False
            
            if result_type == "column":
                item_label = meta.header
            else:
                item_label = meta.first_cell_value
            
            for concept in calc_request.target_concepts:
                concept_lower = concept.lower()
                
                # Get items to check based on result type
                items_to_check = []
                if result_type == "column":
                    # For columns, check both header and sample values
                    items_to_check.append(meta.header.lower())
                    # Also check sample values which may contain the actual concept names
                    for sample_val in meta.sample_values:
                        if isinstance(sample_val, str):
                            items_to_check.append(sample_val.lower())
                else:
                    # For rows, check first cell value
                    items_to_check.append(meta.first_cell_value.lower())
                
                # Check each item for concept match
                for item_name in items_to_check:
                    
                    # Exact match
                    if item_name == concept_lower:
                        is_precise_match = True
                        break
                    
                    # Check if it starts with the concept (but filter out derivatives)
                    if item_name.startswith(concept_lower):
                        # Avoid derivatives like "Revenue Growth" when looking for "Revenue"
                        derivative_words = ["growth", "rate", "ratio", "margin", "percentage", "change", "variance", "forecast", "budget", "plan"]
                        
                        # Get the part after the concept
                        remainder = item_name[len(concept_lower):].strip()
                        
                        # If remainder is empty or just contains acceptable modifiers, it's a match
                        if not remainder or remainder in ["(yr1)", "(yr2)", "(yr3)", "total", "sum"]:
                            is_precise_match = True
                            break
                        
                        # If remainder contains derivative words, skip it
                        if any(word in remainder for word in derivative_words):
                            continue
                        else:
                            is_precise_match = True
                            break
                    
                    # Check if concept is in the name but more carefully
                    if concept_lower in item_name:
                        # Make sure it's not a derivative concept
                        if concept_lower == "revenue":
                            # Exclude things like "Revenue Growth", "Revenue Rate", etc.
                            if any(word in item_name for word in ["growth", "rate", "ratio", "margin", "percentage"]):
                                continue
                        elif concept_lower == "profit":
                            # For profit, be more inclusive but still exclude rates/margins
                            if any(word in item_name for word in ["rate", "ratio", "percentage"]) and "margin" not in item_name:
                                continue
                        
                        is_precise_match = True
                        break
                
                # If we found a match for this concept, break out of concept loop
                if is_precise_match:
                    break
            
            if is_precise_match:
                filtered_results.append((meta, score, result_type))
        
        # Sort by relevance score and return top results
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:10]
    
    def extract_all_column_values(self, column_meta: ColumnMetadata) -> List[float]:
        """Extract ALL numeric values from a column using fresh spreadsheet data"""
        # Fetch fresh data for each calculation
        spreadsheet_data = self._get_fresh_spreadsheet_data()
        
        numeric_values = []
        sheet_name = column_meta.sheet
        
        # Get all cells for this sheet
        if sheet_name not in spreadsheet_data.cells:
            print(f"🔢 DEBUG: Sheet '{sheet_name}' not found in spreadsheet data")
            return numeric_values
        
        sheet_cells = spreadsheet_data.cells[sheet_name]
        
        # Find the column index by looking for the header
        header_cell = None
        for cell in sheet_cells.values():
            if cell.row == 1 and str(cell.value).strip() == column_meta.header:
                header_cell = cell
                break
        
        if not header_cell:
            print(f"🔢 DEBUG: Header '{column_meta.header}' not found in sheet '{sheet_name}'")
            return numeric_values
        
        col_idx = header_cell.col
        print(f"🔢 DEBUG: Found column '{column_meta.header}' at column index {col_idx}")
        
        # Extract all values from this column (skip header row)
        for cell in sheet_cells.values():
            if cell.col == col_idx and cell.row > 1 and cell.value:
                try:
                    if isinstance(cell.value, (int, float)):
                        numeric_values.append(float(cell.value))
                    elif isinstance(cell.value, str):
                        # Clean string values (remove currency symbols, commas, etc.)
                        clean_value = re.sub(r'[,$%]', '', str(cell.value))
                        if clean_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            numeric_values.append(float(clean_value))
                except (ValueError, TypeError):
                    continue
        
        print(f"🔢 DEBUG: Extracted {len(numeric_values)} total values from column '{column_meta.header}': {numeric_values}")
        return numeric_values
    
    def extract_all_row_values(self, row_meta: RowMetadata) -> List[float]:
        """Extract ALL numeric values from a row using fresh spreadsheet data"""
        # Fetch fresh data for each calculation
        spreadsheet_data = self._get_fresh_spreadsheet_data()
        
        numeric_values = []
        sheet_name = row_meta.sheet
        
        # Get all cells for this sheet
        if sheet_name not in spreadsheet_data.cells:
            print(f"🔢 DEBUG: Sheet '{sheet_name}' not found in spreadsheet data")
            return numeric_values
        
        sheet_cells = spreadsheet_data.cells[sheet_name]
        row_idx = row_meta.row_number
        
        print(f"🔢 DEBUG: Extracting values from row {row_idx} in sheet '{sheet_name}'")
        
        # Extract values from this row, but be more selective
        # Only extract from columns that contain data (skip first column which is label)
        row_cells = [cell for cell in sheet_cells.values() if cell.row == row_idx]
        row_cells.sort(key=lambda c: c.col)  # Sort by column order
        
        # Find the range of meaningful data (skip empty columns at the end)
        data_cells = []
        for cell in row_cells:
            if cell.col > 1 and cell.value and str(cell.value).strip():  # Skip first column and empty cells
                # Only include if it looks like numeric data or could be converted
                try:
                    if isinstance(cell.value, (int, float)):
                        data_cells.append(cell)
                    elif isinstance(cell.value, str):
                        clean_value = re.sub(r'[,$%]', '', str(cell.value))
                        if clean_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            data_cells.append(cell)
                except (ValueError, TypeError):
                    continue
        
        # Extract numeric values from the identified data cells
        for cell in data_cells:
            try:
                if isinstance(cell.value, (int, float)):
                    numeric_values.append(float(cell.value))
                elif isinstance(cell.value, str):
                    clean_value = re.sub(r'[,$%]', '', str(cell.value))
                    if clean_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        numeric_values.append(float(clean_value))
            except (ValueError, TypeError):
                continue
        
        print(f"🔢 DEBUG: Extracted {len(numeric_values)} total values from row '{row_meta.first_cell_value}': {numeric_values}")
        return numeric_values
    
    def extract_numeric_values(self, data_sources: List[Tuple[Union[ColumnMetadata, RowMetadata], float, str]]) -> List[float]:
        """Extract numeric values from identified data sources"""
        numeric_values = []
        
        print(f"🔢 DEBUG: Extracting values from {len(data_sources)} data sources")
        
        for meta, score, result_type in data_sources:
            source_values = []
            source_name = meta.header if result_type == "column" else meta.first_cell_value
            
            print(f"🔢 DEBUG: Processing {source_name} ({result_type}) from {meta.sheet}")
            if result_type == "column":
                print(f"🔢 DEBUG: Sample values preview: {meta.sample_values[:3]}")
            else:
                print(f"🔢 DEBUG: Raw sample values: {meta.sample_values}")
            
            if result_type == "column":
                # Extract ALL numeric values from the entire column
                column_values = self.extract_all_column_values(meta)
                source_values.extend(column_values)
                numeric_values.extend(column_values)
            else:  # row
                # Extract ALL numeric values from the entire row
                row_values = self.extract_all_row_values(meta)
                source_values.extend(row_values)
                numeric_values.extend(row_values)
            
            print(f"🔢 DEBUG: Extracted {len(source_values)} values from {source_name}: {source_values}")
        
        print(f"🔢 DEBUG: Total numeric values extracted: {len(numeric_values)} values")
        print(f"🔢 DEBUG: All values: {numeric_values}")
        
        return numeric_values
    
    def perform_calculation(self, calc_request: CalculationRequest, numeric_values: List[float], data_sources: List[Tuple[Union[ColumnMetadata, RowMetadata], float, str]]) -> CalculationResult:
        """Perform the actual calculation"""
        
        if not numeric_values:
            return CalculationResult(
                value=0.0,
                operation=calc_request.operation,
                data_used=[],
                explanation="No numeric data found for calculation",
                formatted_result="Unable to calculate: No numeric data found"
            )
        
        # Perform calculation based on operation
        if calc_request.operation == "sum":
            result_value = sum(numeric_values)
        elif calc_request.operation == "average":
            result_value = sum(numeric_values) / len(numeric_values)
        elif calc_request.operation == "count":
            result_value = len(numeric_values)
        elif calc_request.operation == "min":
            result_value = min(numeric_values)
        elif calc_request.operation == "max":
            result_value = max(numeric_values)
        elif calc_request.operation == "percentage" and len(numeric_values) >= 2:
            # Calculate percentage: first value / second value * 100
            result_value = (numeric_values[0] / numeric_values[1]) * 100 if numeric_values[1] != 0 else 0
        else:
            result_value = sum(numeric_values)  # Default to sum
        
        # Format data sources info
        data_used = []
        for meta, score, result_type in data_sources:
            if result_type == "column":
                data_used.append({
                    "type": "column",
                    "name": meta.header,
                    "sheet": meta.sheet,
                    "range": meta.addresses,
                    "sample_values": meta.sample_values[:3]
                })
            else:
                data_used.append({
                    "type": "row", 
                    "name": meta.first_cell_value,
                    "sheet": meta.sheet,
                    "range": meta.addresses,
                    "sample_values": meta.sample_values[:3]
                })
        
        # Create explanation
        operation_names = {
            "sum": "Sum",
            "average": "Average", 
            "count": "Count",
            "min": "Minimum",
            "max": "Maximum",
            "percentage": "Percentage"
        }
        
        explanation = f"Calculated {operation_names.get(calc_request.operation, calc_request.operation)} of {len(numeric_values)} values from {len(data_sources)} data sources related to: {', '.join(calc_request.target_concepts)}"
        
        # Format result
        if calc_request.operation == "percentage":
            formatted_result = f"{result_value:.2f}%"
        elif calc_request.operation == "count":
            formatted_result = f"{int(result_value)}"
        else:
            formatted_result = f"{result_value:,.2f}"
        
        return CalculationResult(
            value=result_value,
            operation=calc_request.operation,
            data_used=data_used,
            explanation=explanation,
            formatted_result=formatted_result
        )
    
    def calculate(self, query: str) -> CalculationResult:
        """Main method to handle full calculation pipeline"""
        
        # 1. Parse the query
        calc_request = self.parse_calculation_query(query)
        print(f"🔍 Parsed calculation request: {calc_request.operation} on {calc_request.target_concepts}")
        
        # 2. Identify relevant data
        data_sources = self.identify_relevant_data(calc_request)
        print(f"📊 Found {len(data_sources)} relevant data sources")
        
        # 3. Extract numeric values
        numeric_values = self.extract_numeric_values(data_sources)
        print(f"🔢 Extracted {len(numeric_values)} numeric values: {numeric_values[:5]}...")
        
        # 4. Perform calculation
        result = self.perform_calculation(calc_request, numeric_values, data_sources)
        
        return result 