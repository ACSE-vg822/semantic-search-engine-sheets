# calculator_node.py - LangGraph Calculator Node for Computational Queries

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import anthropic
from src.semantic_search.types import SearchState, EnrichedRowData, DataFetchPlan

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalculationResult:
    """Result of a calculation operation"""
    operation: str
    result: Union[float, int, str, List, Dict]
    formula_used: str
    data_summary: Dict[str, Any]
    explanation: str

class CalculatorNode:
    """Node for performing calculations on spreadsheet data"""
    
    def __init__(self, claude_client):
        self.client = claude_client
    
    def __call__(self, state: SearchState) -> SearchState:
        """LangGraph node function for calculations"""
        try:
            # Convert enriched_data back to EnrichedRowData objects
            enriched_data = [self._dict_to_enriched_data(d) for d in state["enriched_data"]]
            
            # Perform calculations
            calculation_result = self.perform_calculations(
                state["query"], 
                enriched_data, 
                state["plan"]
            )
            
            # Store result in state
            state["calculation_result"] = {
                "operation": calculation_result.operation,
                "result": calculation_result.result,
                "formula_used": calculation_result.formula_used,
                "data_summary": calculation_result.data_summary,
                "explanation": calculation_result.explanation
            }
            state["status"] = "calculation_complete"
            
        except Exception as e:
            logger.error(f"Error in calculation: {e}")
            state["calculation_result"] = {
                "operation": "error",
                "result": None,
                "formula_used": "",
                "data_summary": {},
                "explanation": f"Calculation failed: {str(e)}"
            }
            state["status"] = "error"
        
        return state
    
    def perform_calculations(self, query: str, enriched_data: List[EnrichedRowData], plan: Dict) -> CalculationResult:
        """Main calculation logic with LLM assistance for formula suggestions"""
        
        # Convert data to pandas DataFrame for easier manipulation
        df_data = self._convert_to_dataframe(enriched_data)
        
        # Get calculation strategy from LLM
        calculation_strategy = self._get_calculation_strategy(query, df_data, plan)
        
        # Execute the calculation
        result = self._execute_calculation(calculation_strategy, df_data)
        
        return CalculationResult(
            operation=calculation_strategy.get("operation", "unknown"),
            result=result["value"],
            formula_used=result["formula"],
            data_summary=result["data_summary"],
            explanation=result["explanation"]
        )
    
    def _convert_to_dataframe(self, enriched_data: List[EnrichedRowData]) -> pd.DataFrame:
        """Convert enriched data to pandas DataFrame"""
        
        if not enriched_data:
            return pd.DataFrame()
        
        # Collect all data into a structured format
        all_data = []
        
        for data in enriched_data:
            if data.values:
                # Create a record with metadata
                record = {
                    "sheet": data.sheet,
                    "source": data.first_cell_value,
                    "row_number": data.row_number,
                    "cell_addresses": data.cell_addresses
                }
                
                # Add values with their headers if available
                if data.headers and len(data.headers) == len(data.values):
                    for header, value in zip(data.headers, data.values):
                        # Try to convert to numeric if possible
                        try:
                            record[header] = pd.to_numeric(value)
                        except (ValueError, TypeError):
                            record[header] = value
                else:
                    # Add values with generic column names
                    for i, value in enumerate(data.values):
                        try:
                            record[f"value_{i}"] = pd.to_numeric(value)
                        except (ValueError, TypeError):
                            record[f"value_{i}"] = value
                
                all_data.append(record)
        
        return pd.DataFrame(all_data)
    
    def _get_calculation_strategy(self, query: str, df: pd.DataFrame, plan: Dict) -> Dict:
        """Use LLM to determine calculation strategy"""
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(df)
        
        system_prompt = """You are a spreadsheet calculation assistant. Given a user query and available data, 
        determine the appropriate calculation strategy.

        Respond with a JSON object containing:
        {
            "operation": "sum|max|min|average|count|percentage|growth_rate|custom",
            "target_columns": ["column1", "column2"],
            "filters": {"column": "value"},
            "formula": "pandas/numpy expression or mathematical formula",
            "explanation": "natural language explanation of what will be calculated"
        }
        
        Common operations:
        - sum: Total of numeric values
        - max/min: Maximum/minimum values
        - average: Mean of numeric values
        - count: Count of non-null values
        - percentage: Percentage calculations
        - growth_rate: Period-over-period growth
        - custom: Complex formula requiring pandas operations
        
        For formulas, use pandas syntax like: df['column'].sum(), df['column'].max(), etc.
        """
        
        user_prompt = f"""
        User Query: "{query}"
        
        Available Data Summary:
        {data_summary}
        
        What calculation should be performed? Provide a specific pandas formula.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=800,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            strategy = json.loads(response.content[0].text)
            return strategy
            
        except Exception as e:
            logger.error(f"Error getting calculation strategy: {e}")
            # Fallback strategy
            return {
                "operation": "sum",
                "target_columns": [],
                "filters": {},
                "formula": "df.select_dtypes(include=[np.number]).sum().sum()",
                "explanation": "Sum of all numeric values"
            }
    
    def _execute_calculation(self, strategy: Dict, df: pd.DataFrame) -> Dict:
        """Execute the calculation based on strategy"""
        
        try:
            # Apply filters if specified
            filtered_df = df.copy()
            if strategy.get("filters"):
                for column, value in strategy["filters"].items():
                    if column in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df[column] == value]
            
            # Execute the formula
            formula = strategy.get("formula", "")
            if formula:
                # Create safe execution environment
                safe_locals = {
                    "df": filtered_df,
                    "pd": pd,
                    "np": np
                }
                
                # Execute the formula
                result_value = eval(formula, {"__builtins__": {}}, safe_locals)
                
                # Handle different result types
                if isinstance(result_value, (pd.Series, pd.DataFrame)):
                    if len(result_value) == 1:
                        result_value = result_value.iloc[0]
                    else:
                        result_value = result_value.to_dict()
                
            else:
                result_value = "No formula provided"
            
            # Create data summary
            data_summary = {
                "rows_processed": len(filtered_df),
                "columns_used": strategy.get("target_columns", []),
                "numeric_columns": list(filtered_df.select_dtypes(include=[np.number]).columns),
                "operation_type": strategy.get("operation", "unknown")
            }
            
            return {
                "value": result_value,
                "formula": formula,
                "data_summary": data_summary,
                "explanation": strategy.get("explanation", "Calculation completed")
            }
            
        except Exception as e:
            logger.error(f"Error executing calculation: {e}")
            return {
                "value": None,
                "formula": strategy.get("formula", ""),
                "data_summary": {"error": str(e)},
                "explanation": f"Calculation failed: {str(e)}"
            }
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare a summary of available data for LLM"""
        if df.empty:
            return "No data available"
        
        summary_parts = [
            f"Dataset: {len(df)} rows, {len(df.columns)} columns",
            f"Columns: {list(df.columns)}",
        ]
        
        # Add numeric column info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            summary_parts.append(f"Numeric columns: {numeric_cols}")
            
            # Add sample statistics for numeric columns
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                stats = df[col].describe()
                summary_parts.append(f"{col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}")
        
        # Add sample data
        if len(df) > 0:
            summary_parts.append("Sample data (first 3 rows):")
            sample_data = df.head(3).to_dict('records')
            for i, row in enumerate(sample_data):
                summary_parts.append(f"  Row {i+1}: {row}")
        
        return "\n".join(summary_parts)
    
    def _dict_to_enriched_data(self, data_dict: Dict) -> EnrichedRowData:
        """Convert dictionary back to EnrichedRowData object"""
        return EnrichedRowData(
            first_cell_value=data_dict["first_cell_value"],
            sheet=data_dict["sheet"],
            row_number=data_dict["row_number"],
            values=data_dict["values"],
            formulas=data_dict["formulas"],
            headers=data_dict["headers"],
            cell_addresses=data_dict["cell_addresses"],
            cross_references=data_dict["cross_references"],
            metadata=data_dict["metadata"]
        ) 