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
                # Create a record with metadata - ensure these are NOT numeric columns
                record = {
                    "sheet": str(data.sheet),  # Force string type
                    "source": str(data.first_cell_value),  # Force string type
                    "row_number_meta": str(data.row_number),  # Rename and force string to avoid numeric treatment
                    "cell_addresses": str(data.cell_addresses)  # Force string type
                }
                
                # Add values with their headers if available
                if data.headers and len(data.headers) == len(data.values):
                    for header, value in zip(data.headers, data.values):
                        # Filter out empty/null values before conversion
                        if value is not None and value != "" and value != " ":
                            try:
                                numeric_value = pd.to_numeric(value)
                                # Only include if it's a valid number (not NaN)
                                if not pd.isna(numeric_value):
                                    record[header] = numeric_value
                                else:
                                    record[header] = value
                            except (ValueError, TypeError):
                                record[header] = value
                        else:
                            # Skip empty values entirely
                            record[header] = pd.NA
                else:
                    # Add values with generic column names
                    for i, value in enumerate(data.values):
                        # Filter out empty/null values before conversion
                        if value is not None and value != "" and value != " ":
                            try:
                                numeric_value = pd.to_numeric(value)
                                # Only include if it's a valid number (not NaN)
                                if not pd.isna(numeric_value):
                                    record[f"value_{i}"] = numeric_value
                                else:
                                    record[f"value_{i}"] = value
                            except (ValueError, TypeError):
                                record[f"value_{i}"] = value
                        else:
                            # Skip empty values entirely
                            record[f"value_{i}"] = pd.NA
                
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
        - sum: Total of numeric values - use df.select_dtypes(include=[np.number]).sum().sum() for all numeric values
        - max/min: Maximum/minimum values - use df.select_dtypes(include=[np.number]).max().max() or .min().min()
        - average: Mean of numeric values - use df.select_dtypes(include=[np.number]).mean().mean()
        - count: Count of non-null values
        - percentage: Percentage calculations
        - growth_rate: Period-over-period growth
        - custom: Complex formula requiring pandas operations
        
        For formulas, use pandas syntax like: 
        - df['column'].sum() for single column sum
        - df.select_dtypes(include=[np.number]).sum().sum() for sum of all numeric values
        - df['column'].max(), df['column'].min(), df['column'].mean() for specific columns
        - df.select_dtypes(include=[np.number]).max().max() for maximum of all numeric values
        - df.select_dtypes(include=[np.number]).min().min() for minimum of all numeric values
        - df.select_dtypes(include=[np.number]).mean().mean() for average of all numeric values
        
        IMPORTANT: 
        - For aggregate operations (sum, max, min, mean), always use double aggregation (.sum().sum(), .max().max(), etc.) to get a single scalar value, not a Series.
        - Pandas automatically excludes NA/NaN values from min/max/mean calculations
        - Always use skipna=True (default) to ignore missing values
        - Exclude metadata columns (sheet, source, row_number_meta, cell_addresses) from calculations
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
            # Intelligent fallback strategy based on query keywords
            query_lower = query.lower()
            
            # Define metadata columns to exclude from calculations
            metadata_columns = ["sheet", "source", "row_number_meta", "cell_addresses"]
            
            if any(keyword in query_lower for keyword in ["max", "maximum", "highest", "largest"]):
                operation = "max"
                formula = f"df.drop(columns={metadata_columns}, errors='ignore').select_dtypes(include=[np.number]).max(skipna=True).max(skipna=True)"
                explanation = "Maximum of all numeric values (excluding empty cells and metadata)"
            elif any(keyword in query_lower for keyword in ["min", "minimum", "lowest", "smallest"]):
                operation = "min"
                formula = f"df.drop(columns={metadata_columns}, errors='ignore').select_dtypes(include=[np.number]).min(skipna=True).min(skipna=True)"
                explanation = "Minimum of all numeric values (excluding empty cells and metadata)"
            elif any(keyword in query_lower for keyword in ["average", "mean", "avg"]):
                operation = "average"
                formula = f"df.drop(columns={metadata_columns}, errors='ignore').select_dtypes(include=[np.number]).mean(skipna=True).mean(skipna=True)"
                explanation = "Average of all numeric values (excluding empty cells and metadata)"
            else:
                operation = "sum"
                formula = f"df.drop(columns={metadata_columns}, errors='ignore').select_dtypes(include=[np.number]).sum(skipna=True).sum(skipna=True)"
                explanation = "Sum of all numeric values (excluding empty cells and metadata)"
            
            return {
                "operation": operation,
                "target_columns": [],
                "filters": {},
                "formula": formula,
                "explanation": explanation
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
                if isinstance(result_value, pd.Series):
                    # For Series, check if it's a single value or multiple values
                    if len(result_value) == 1:
                        result_value = result_value.iloc[0]
                    else:
                        # If multiple values, aggregate them based on operation type
                        operation = strategy.get("operation", "").lower()
                        if operation in ["sum", "total"]:
                            result_value = result_value.sum()
                        elif operation in ["max", "maximum"]:
                            result_value = result_value.max()
                        elif operation in ["min", "minimum"]:
                            result_value = result_value.min()
                        elif operation in ["mean", "average", "avg"]:
                            result_value = result_value.mean()
                        else:
                            result_value = result_value.to_dict()
                elif isinstance(result_value, pd.DataFrame):
                    # For DataFrame, convert to dict or extract single value
                    if result_value.size == 1:
                        result_value = result_value.iloc[0, 0]
                    else:
                        result_value = result_value.to_dict()
                
            else:
                result_value = "No formula provided"
            
            # Create data summary excluding metadata columns
            metadata_columns = ["sheet", "source", "row_number_meta", "cell_addresses"]
            numeric_df = filtered_df.drop(columns=metadata_columns, errors='ignore')
            
            data_summary = {
                "rows_processed": len(filtered_df),
                "columns_used": strategy.get("target_columns", []),
                "numeric_columns": list(numeric_df.select_dtypes(include=[np.number]).columns),
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
        
        # Exclude metadata columns from numeric analysis
        metadata_columns = ["sheet", "source", "row_number_meta", "cell_addresses"]
        numeric_df = df.drop(columns=metadata_columns, errors='ignore')
        numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            summary_parts.append(f"Numeric columns: {numeric_cols}")
            
            # Add sample statistics for numeric columns
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                stats = numeric_df[col].describe()
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