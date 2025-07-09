# src/rag/query_engine.py

import os
import json
import toml
import logging
import anthropic
from typing import List, Dict, Any, Union
from dataclasses import asdict

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import gspread
from google.oauth2 import service_account

from src.rag.retriever import SpreadsheetRetriever
from src.data_ingestion.spreadsheet_parser_advance import ColumnMetadata, RowMetadata, SpreadsheetKnowledgeGraph

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, retriever: SpreadsheetRetriever, spreadsheet_id: str = None):
        self.retriever = retriever
        self.spreadsheet_id = spreadsheet_id
        self.api_key = self._load_claude_key()
        if not self.api_key:
            raise ValueError("Claude API key not found in Streamlit secrets or local secrets.toml.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

        if self.spreadsheet_id:
            self.sheets_client = self._setup_google_sheets_auth()

    def _setup_google_sheets_auth(self):
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        return gspread.authorize(credentials)

    def _load_claude_key(self) -> str:
        if STREAMLIT_AVAILABLE:
            try:
                return st.secrets["claude_api_key"]
            except Exception as e:
                logger.debug(f"Streamlit secrets unavailable: {e}")

        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            secrets_path = os.path.join(project_root, '.streamlit', 'secrets.toml')
            if os.path.exists(secrets_path):
                secrets = toml.load(secrets_path)
                return secrets.get("claude_api_key", "")
        except Exception as e:
            logger.warning(f"Error loading local secrets.toml: {e}")
        return ""

    def _fetch_column_data(self, sheet_name: str, cell_range: str) -> List[Union[str, float, int]]:
        if not self.spreadsheet_id or not self.sheets_client:
            return []

        try:
            spreadsheet = self.sheets_client.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            values = worksheet.get(cell_range, value_render_option='UNFORMATTED_VALUE')

            column_values = []
            for row in values:
                for cell in row:
                    if cell not in [None, "", " "]:
                        column_values.append(cell)
            return column_values
        except Exception as e:
            logger.warning(f"Error fetching data for {sheet_name}!{cell_range}: {e}")
            return []

    def _calculate_numerical_stats(self, values: List[Union[str, float, int]]) -> Dict[str, Any]:
        stats = []
        numeric = []
        for v in values:
            if isinstance(v, (int, float)):
                numeric.append(v)
            elif isinstance(v, str):
                try:
                    numeric.append(float(v))
                except ValueError:
                    continue

        if numeric:
            return {
                "min": min(numeric),
                "max": max(numeric),
                "mean": sum(numeric) / len(numeric),
                "count": len(numeric),
                "total_cells": len(values)
            }
        return {}

    def _enhance_columns_with_stats(self, columns: List[ColumnMetadata]) -> List[Dict[str, Any]]:
        enriched = []
        for col in columns:
            entry = {
                "sheet": col.sheet,
                "header": col.header,
                "data_type": col.data_type,
                "cell_range": col.addresses,
                "formula": col.first_cell_formula,
                "cross_sheet_refs": col.cross_sheet_refs,
                "sample_values": col.sample_values,
            }
            if col.data_type in ['number', 'formula']:
                if col.addresses:
                    values = self._fetch_column_data(col.sheet, col.addresses)
                    stats = self._calculate_numerical_stats(values)
                    if stats:
                        entry["statistics"] = stats
            enriched.append(entry)
        return enriched

    def _build_context(self, columns: List[ColumnMetadata], rows: List[RowMetadata]) -> str:
        context = {
            "columns": self._enhance_columns_with_stats(columns),
            "rows": [asdict(r) for r in rows]
        }
        return json.dumps(context, indent=2)

    def ask(self, query: str) -> str:
        top_results = self.retriever.retrieve(query, top_k=10)  # Increase top_k to get more results
        
        # Debug: Print what we found
        print(f"\n🔍 Debug: Found {len(top_results)} total results")
        for metadata, score, entry_type in top_results:
            if entry_type == "row":
                print(f"  Row: {metadata.concept} (score: {score:.3f})")
            elif entry_type == "column":
                print(f"  Column: {metadata.header} (score: {score:.3f})")
        
        # Separate columns and rows from the results
        col_metadata = [metadata for metadata, score, entry_type in top_results if entry_type == "column"][:3]
        row_metadata = [metadata for metadata, score, entry_type in top_results if entry_type == "row"][:5]  # Increase to 5 rows
        
        print(f"\n📊 Using {len(col_metadata)} columns and {len(row_metadata)} rows in context")

        context = self._build_context(col_metadata, row_metadata)

        system_prompt = self._get_enhanced_system_prompt()

        user_prompt = f"User query: {query}\n\nSpreadsheet context:\n{context}\n\nIMPORTANT: The user asked to '{query}'. Please analyze and return information about ALL relevant rows/entries found in the context above, not just a subset. Each row represents a data entry that should be explained."

        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=8192,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text

    def _get_enhanced_system_prompt(self) -> str:
        return """You are an advanced spreadsheet AI assistant with deep business intelligence capabilities. 
Given a user query and structured spreadsheet context, provide a helpful, precise JSON-formatted answer.

## CORE CAPABILITIES:

### 1. CONCEPT RECOGNITION
Identify and understand business concepts including:
- **Financial**: revenue, sales, income, costs, expenses, profit, earnings, margins, ratios, ROI, ROE, EBITDA
- **Operational**: productivity, efficiency, utilization, capacity, throughput, quality metrics
- **Temporal**: forecasts, projections, targets, budgets, actuals, variance, growth rates
- **Performance**: KPIs, benchmarks, trends, comparisons, rankings

### 2. SYNONYM HANDLING
Understand semantic equivalencies:
- sales = revenue = income = turnover
- profit = earnings = net income = bottom line
- costs = expenses = expenditures = outlays
- efficiency = productivity = performance = utilization
- margin = markup = spread
- forecast = projection = prediction = estimate
- target = goal = objective = budget

### 3. CONTEXT INTERPRETATION
Analyze formulas and their business meaning:
- **Division formulas** (=A/B) in "Margin", "Ratio", "%" columns → percentage/ratio calculations
- **SUM formulas** in "Total", "Sum", "Aggregate" columns → accumulation of values
- **Subtraction** (=A-B) in "Variance", "Difference", "Gap" columns → performance vs target
- **Growth formulas** ((New-Old)/Old) → percentage change calculations
- **Average/Mean** calculations → central tendency metrics

### 4. FORMULA SEMANTICS
Interpret formulas based on column context:
- =SUM() in "Total Sales" → revenue aggregation
- =B5/B6 in "Margin %" → profitability ratio
- =(Current-Previous)/Previous in "Growth" → percentage change
- =AVERAGE() in performance columns → mean performance metric
- =IF statements → conditional business logic

### 5. BUSINESS INTELLIGENCE
Provide insights by:
- Identifying trends and patterns
- Explaining business significance of metrics
- Contextualizing numbers within business operations
- Highlighting relationships between different metrics

## OUTPUT FORMAT:
Provide responses as JSON array with enhanced business context:
[{
  "sheet": "Sheet name",
  "header": "Column header",
  "cell_range": "Cell addresses",
  "formula": "Formula if any",
  "explanation": "Detailed explanation with business context"
}]

## INSTRUCTIONS:
- IMPORTANT: DO NOT HALLUCINATE ANY INFORMATION. ONLY USE THE INFORMATION PROVIDED IN THE CONTEXT.
- Always interpret data through a business lens
- Recognize synonyms and related concepts in queries
- Explain the business significance of formulas and calculations
- When users ask for "all" items, return ALL relevant matches
- Connect data points to broader business insights
- Use business terminology appropriately
"""


# 🧪 Test block
if __name__ == "__main__":
    from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced

    TEST_SPREADSHEET_ID = "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"

    parser = SpreadsheetParserAdvanced()
    spreadsheet = parser.parse(TEST_SPREADSHEET_ID)
    kg = parser.build_knowledge_graph(spreadsheet)
    retriever = SpreadsheetRetriever(kg, use_embeddings=True)

    engine = QueryEngine(retriever, spreadsheet_id=TEST_SPREADSHEET_ID)

    query = input("🔍 Enter query: ")
    response = engine.ask(query)
    print("\n📥 Claude's Response:\n")
    print(response)