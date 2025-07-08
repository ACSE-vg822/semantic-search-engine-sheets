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
from src.data_ingestion.spreadsheet_parser_advance import ColumnMetadata

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, retriever: SpreadsheetRetriever, spreadsheet_id: str = None):
        self.retriever = retriever
        self.spreadsheet_id = spreadsheet_id
        self.api_key = self._load_claude_key()
        if not self.api_key:
            raise ValueError("Claude API key not found in Streamlit secrets or local secrets.toml.")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Setup Google Sheets client for on-demand data fetching
        if self.spreadsheet_id:
            self.sheets_client = self._setup_google_sheets_auth()

    def _setup_google_sheets_auth(self):
        """Setup Google Sheets authentication."""
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        return gspread.authorize(credentials)

    def _load_claude_key(self) -> str:
        """Load Claude API key from Streamlit secrets or local .streamlit/secrets.toml."""
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
        """Fetch actual column data from Google Sheets for statistical analysis."""
        if not self.spreadsheet_id or not self.sheets_client:
            return []
        
        try:
            spreadsheet = self.sheets_client.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            
            # Fetch the specific range
            values = worksheet.get(cell_range, value_render_option='UNFORMATTED_VALUE')
            
            # Flatten the values list and filter out empty cells
            column_values = []
            for row in values:
                if row:  # Skip empty rows
                    for cell in row:
                        if cell not in [None, "", " "]:
                            column_values.append(cell)
            
            return column_values
            
        except Exception as e:
            logger.warning(f"Error fetching column data for {sheet_name}!{cell_range}: {e}")
            return []

    def _calculate_numerical_stats(self, values: List[Union[str, float, int]]) -> Dict[str, Any]:
        """Calculate min, max, mean for numerical values."""
        stats = {}
        
        # Filter numerical values
        numerical_values = []
        for value in values:
            if isinstance(value, (int, float)):
                numerical_values.append(float(value))
            elif isinstance(value, str):
                try:
                    # Try to convert string to number
                    numerical_values.append(float(value))
                except (ValueError, TypeError):
                    continue
        
        if numerical_values:
            stats['min'] = min(numerical_values)
            stats['max'] = max(numerical_values)
            stats['mean'] = sum(numerical_values) / len(numerical_values)
            stats['count'] = len(numerical_values)
            stats['total_cells'] = len(values)
            
        return stats

    def _enhance_columns_with_stats(self, columns: List[ColumnMetadata]) -> List[Dict[str, Any]]:
        """Enhance column metadata with statistical information by fetching actual data."""
        enhanced_columns = []
        
        for col in columns:
            col_dict = {
                'sheet': col.sheet,
                'header': col.header,
                'data_type': col.data_type,
                'cell_range': col.addresses,
                'formula': col.first_cell_formula,
                'cross_sheet_refs': col.cross_sheet_refs,
                'sample_values': col.sample_values,
            }
            
            # Fetch actual column data and calculate statistics for numerical columns
            if col.data_type in ['number', 'formula'] or any(isinstance(v, (int, float)) for v in col.sample_values):
                if col.addresses:  # Make sure we have a valid range
                    actual_values = self._fetch_column_data(col.sheet, col.addresses)
                    if actual_values:
                        stats = self._calculate_numerical_stats(actual_values)
                        if stats:  # Only add stats if we found numerical values
                            col_dict['statistics'] = stats
                            logger.info(f"Added stats for column '{col.header}' in '{col.sheet}': {stats}")
            
            enhanced_columns.append(col_dict)
        
        return enhanced_columns

    def _build_context(self, columns: List[ColumnMetadata]) -> str:
        """Build context with enhanced statistical information."""
        enhanced_columns = self._enhance_columns_with_stats(columns)
        return json.dumps(enhanced_columns, indent=2)

    def ask(self, query: str) -> str:
        # Retrieve top 5 relevant columns
        top_columns = self.retriever.retrieve(query, top_k=5)
        top_metadata = [col for col, _ in top_columns]

        # Build enhanced context with fresh statistical analysis
        context = self._build_context(top_metadata)

        system_prompt = (
            "You are a spreadsheet AI assistant. You are exceptionally adept at analyzing business oriented spreadsheets. Given a user query and structured spreadsheet context, "
            "provide a precise and helpful answer using only the context provided.\n\n"
            "The context includes statistical information (min, max, mean, count) for numerical columns when available. "
            "When describing numerical ranges, use clear, consistent formatting:\n"
            "- For ranges: 'ranging from X to Y' (not X/Y or X₨Y)\n"
            "- Use consistent currency symbols or no currency symbols. If currency is not specified, don't hallucinate it\n"
            "- Format numbers clearly with commas: 18,000 to 36,000\n"
            "- When mentioning averages, say 'with an average of X'\n\n"
            "Your response must follow this JSON format:\n"
            "[{ concept_name, sheet, header, cell_range, formula, explanation }]"
        )

        user_prompt = f"""User query: {query}

Spreadsheet context:
{context}
"""

        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text


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
