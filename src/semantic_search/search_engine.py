# src/semantic_search/search_engine.py

import os
import toml
import logging
import json
from typing import List, Dict, Any

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import anthropic
from src.data_ingestion.spreadsheet_parser import SpreadsheetData, CellInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SemanticSearchEngine:
    def __init__(self):
        self.api_key = self._load_claude_api_key()
        if not self.api_key:
            raise ValueError("Claude API key not found in Streamlit secrets or local secrets.toml.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _load_claude_api_key(self) -> str:
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

    def extract_semantic_chunks(self, spreadsheet: SpreadsheetData) -> List[Dict[str, Any]]:
        summaries = []

        for sheet_name, cell_map in spreadsheet.cells.items():
            for cell_addr, cell in cell_map.items():
                header = self._find_header(cell, spreadsheet, sheet_name)

                summaries.append({
                    "sheet": sheet_name,
                    "cell": cell_addr,
                    "header": header,
                    "formula": cell.formula if cell.formula else None,
                    "value": cell.value,
                    "data_type": cell.data_type
                })

        return summaries


    def _find_header(self, cell: CellInfo, spreadsheet: SpreadsheetData, sheet_name: str) -> str:
        """Find the header text for a given cell by scanning upwards in the same column."""
        top_row = spreadsheet.cells[sheet_name]
        for row in range(cell.row - 1, 0, -1):
            candidate_addr = self._cell_address(row, cell.col)
            if candidate_addr in top_row and top_row[candidate_addr].data_type == "text":
                return top_row[candidate_addr].value
        return ""

    def _cell_address(self, row: int, col: int) -> str:
        """Convert row, col to Excel-like cell address (e.g., A1, B3)."""
        col_letter = ""
        while col > 0:
            col -= 1
            col_letter = chr(col % 26 + ord("A")) + col_letter
            col //= 26
        return f"{col_letter}{row}"

    def search(self, query: str, summaries: List[Dict[str, Any]]) -> str:
        """Send a natural language query and spreadsheet summaries to Claude."""
        system_prompt = (
            "You are a financial analyst AI assistant. "
            "Given a user query and spreadsheet summaries, return the most semantically relevant entries. "
            "Each summary includes the header, formula, and location. "
            "Explain why each one matches."
        )

        # Convert summaries to readable JSON
        summaries_text = json.dumps(summaries, indent=2)

        user_prompt = f"""
Query: {query}

Summaries:
{summaries_text}

- Respond in JSON list with keys: concept_name, sheet, cell, header, formula, explanation.
- If the query matches with a whole row or column then ALWAYS insert cell range as the value of cell key.
    Example:
    "cell": "A1:A100"
- If the most relevant entries don't have formulas, you can insert null values for the formula key.
"""

        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        return ''.join(block.text for block in response.content if block.type == "text")
