# spreadsheet_parser.py (with header column detection restricted to Column A)

import gspread
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from google.oauth2 import service_account
import streamlit as st
import os
import re
from datetime import datetime, timedelta
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Data Classes ===

@dataclass
class CellInfo:
    value: Any
    formula: Optional[str] = None
    data_type: str = "unknown"
    formatting: Dict[str, Any] = field(default_factory=dict)
    row: int = 0
    col: int = 0
    address: str = ""
    is_header: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SheetStructure:
    name: str
    total_rows: int
    total_cols: int
    data_ranges: List[Tuple[int, int, int, int]] = field(default_factory=list)
    header_rows: List[int] = field(default_factory=list)
    header_cols: List[int] = field(default_factory=list)
    table_regions: List[Dict[str, Any]] = field(default_factory=list)
    business_domain: Optional[str] = None

@dataclass
class SpreadsheetData:
    spreadsheet_id: str
    title: str
    sheets: Dict[str, SheetStructure]
    cells: Dict[str, Dict[str, CellInfo]]
    cross_sheet_references: Dict[str, List[str]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


# === Spreadsheet Parser ===

class SpreadsheetParser:

    def __init__(self):
        self.gc = self._setup_google_sheets_auth()

    def _setup_google_sheets_auth(self):
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        return gspread.authorize(credentials)

    def parse_spreadsheet(self, spreadsheet_id: str) -> SpreadsheetData:
        spreadsheet = self.gc.open_by_key(spreadsheet_id)
        spreadsheet_data = SpreadsheetData(
            spreadsheet_id=spreadsheet_id,
            title=spreadsheet.title,
            sheets={},
            cells={}
        )

        for sheet in spreadsheet.worksheets():
            sheet_data = self._parse_sheet(sheet)
            spreadsheet_data.sheets[sheet.title] = sheet_data
            spreadsheet_data.cells[sheet.title] = self._extract_cells(sheet)

        spreadsheet_data.cross_sheet_references = self._find_cross_sheet_references(spreadsheet_data)
        return spreadsheet_data

    def _parse_sheet(self, sheet) -> SheetStructure:
        values = sheet.get_all_values()
        formulas = sheet.get_all_values(value_render_option='FORMULA')

        structure = SheetStructure(
            name=sheet.title,
            total_rows=sheet.row_count,
            total_cols=sheet.col_count,
            header_rows=self._detect_header_rows(values),
            header_cols=self._detect_header_cols(values),
            data_ranges=self._detect_data_ranges(values),
            table_regions=self._detect_table_regions(values, formulas),
            business_domain=self._detect_business_domain(values)
        )
        return structure

    def _extract_cells(self, sheet) -> Dict[str, CellInfo]:
        values = sheet.get_all_values()
        formulas = sheet.get_all_values(value_render_option='FORMULA')
        cells = {}

        for row_idx, (val_row, formula_row) in enumerate(zip(values, formulas)):
            for col_idx, (val, formula) in enumerate(zip(val_row, formula_row)):
                if val or formula:
                    addr = self._get_cell_address(row_idx + 1, col_idx + 1)
                    formula_str = str(formula)
                    has_formula = isinstance(formula_str, str) and formula_str.strip().startswith("=")

                    cells[addr] = CellInfo(
                        value=val,
                        formula=formula_str if has_formula else None,
                        data_type=self._detect_data_type(val, formula_str),
                        row=row_idx + 1,
                        col=col_idx + 1,
                        address=addr,
                        is_header=self._is_header_cell(row_idx + 1, col_idx + 1, values)
                    )
        return cells

    def _detect_data_type(self, value: str, formula: str) -> str:
        formula_str = str(formula) if formula is not None else ""

        if formula_str.strip().startswith('='):
            return "formula"

        value_str = str(value).strip()
        if not value_str:
            return "empty"

        try:
            float_val = float(value_str.replace(',', '').replace('%', '').replace('$', ''))
            if 40000 < float_val < 50000:
                return "date"
            return "number"
        except ValueError:
            pass

        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value_str):
            return "date"
        if re.match(r'\d{4}[-/]\d{1,2}$', value_str):
            return "date"
        if value_str.lower() in ['true', 'false', 'yes', 'no']:
            return "boolean"

        return "text"

    def _detect_header_rows(self, values: List[List[str]]) -> List[int]:
        header_rows = []
        for i in range(min(10, len(values) - 1)):
            row, next_row = values[i], values[i + 1]
            if (self._is_mostly_text(row) and self._is_mostly_number_or_formula(next_row)):
                header_rows.append(i + 1)
        return header_rows

    def _detect_header_cols(self, values: List[List[str]]) -> List[int]:
        if not values or len(values) < 2:
            return []

        first_col = [row[0] for row in values if len(row) > 0]
        if len(first_col) < 2:
            return []

        is_first_text = self._detect_data_type(first_col[0], "") == "text"
        rest_are_text = all(self._detect_data_type(cell, "") == "text" for cell in first_col[1:5] if cell)

        if is_first_text and rest_are_text:
            return [1]  # Column A
        return []

    def _detect_data_ranges(self, values: List[List[str]]) -> List[Tuple[int, int, int, int]]:
        non_empty_rows = [i for i, row in enumerate(values) if any(cell.strip() for cell in row)]
        if not non_empty_rows:
            return []
        start_row, end_row = non_empty_rows[0], non_empty_rows[-1]
        max_cols = max(len(row) for row in values)
        start_col, end_col = 0, max_cols - 1
        return [(start_row + 1, end_row + 1, start_col + 1, end_col + 1)]

    def _detect_table_regions(self, values: List[List[str]], formulas: List[List[str]]) -> List[Dict[str, Any]]:
        table_regions = []
        max_row = len(values)
        i = 0
        while i < max_row - 1:
            current_row = values[i]
            next_row = values[i + 1]

            if (any(cell.strip() for cell in current_row) and
                self._is_mostly_text(current_row) and
                self._is_mostly_number_or_formula(next_row)):

                header_row_idx = i
                data_start_row_idx = i + 1
                end_row_idx = data_start_row_idx
                while (end_row_idx < max_row and 
                       any(cell.strip() for cell in values[end_row_idx])):
                    end_row_idx += 1

                column_count = sum(1 for cell in current_row if cell.strip())

                table_regions.append({
                    'start_row': header_row_idx + 1,
                    'header_row': header_row_idx + 1,
                    'data_start_row': data_start_row_idx + 1,
                    'estimated_end_row': end_row_idx,
                    'columns': column_count
                })
                i = end_row_idx
            else:
                i += 1

        return table_regions

    def _detect_business_domain(self, values: List[List[str]]) -> Optional[str]:
        BUSINESS_DOMAINS = {
            'finance': ['revenue', 'cost', 'profit', 'margin', 'ebitda'],
            'sales': ['sales', 'customer', 'quota', 'lead'],
            'operations': ['efficiency', 'capacity', 'throughput']
        }
        content = ' '.join(' '.join(row) for row in values).lower()
        scores = {domain: sum(1 for kw in kws if kw in content) for domain, kws in BUSINESS_DOMAINS.items()}
        return max(scores, key=scores.get) if scores else None

    def _get_cell_address(self, row: int, col: int) -> str:
        col_letter = ""
        while col > 0:
            col -= 1
            col_letter = chr(col % 26 + ord('A')) + col_letter
            col //= 26
        return f"{col_letter}{row}"

    def _is_mostly_text(self, cells: List[str]) -> bool:
        return sum(1 for c in cells if self._detect_data_type(c, "") == "text") / len(cells) > 0.7

    def _is_mostly_number_or_formula(self, cells: List[str]) -> bool:
        return sum(1 for c in cells if self._detect_data_type(c, "") in ["number", "formula"]) / len(cells) > 0.3

    def _is_header_cell(self, row: int, col: int, values: List[List[str]]) -> bool:
        return row <= 3 or col <= 2

    def _find_cross_sheet_references(self, spreadsheet_data: SpreadsheetData) -> Dict[str, List[str]]:
        refs = {}
        for sheet_name, cells in spreadsheet_data.cells.items():
            found = set()
            for cell in cells.values():
                if cell.formula:
                    found.update(re.findall(r"'?([A-Za-z0-9 _-]+)'?!", cell.formula))
            refs[sheet_name] = list(found - {sheet_name})
        return refs



if __name__ == "__main__":
    import streamlit as st
    from pprint import pprint

    # Replace this with your actual spreadsheet ID
    TEST_SPREADSHEET_ID = "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"

    parser = SpreadsheetParser()
    try:
        spreadsheet_data = parser.parse_spreadsheet(TEST_SPREADSHEET_ID)

        print(f"\n📊 Spreadsheet Title: {spreadsheet_data.title}")
        print(f"📄 Total Sheets: {len(spreadsheet_data.sheets)}\n")

        for sheet_name, sheet in spreadsheet_data.sheets.items():
            print(f"--- Sheet: {sheet_name} ---")
            print(f"Size: {sheet.total_rows} rows x {sheet.total_cols} cols")
            print(f"Header Rows: {sheet.header_rows}")
            print(f"Header Columns: {sheet.header_cols}")
            print(f"Data Ranges: {sheet.data_ranges}")
            print(f"Business Domain: {sheet.business_domain}")
            print(f"Table Regions: {sheet.table_regions}\n")

            sample_cells = spreadsheet_data.cells.get(sheet_name, {})
            print("🔍 Sample Cells:")
            for addr, cell in list(sample_cells.items())[:10]:
                print(f"  {addr}: {cell.value} | type: {cell.data_type} | formula: {cell.formula}")

            print("\n")

        print("✅ Parsing complete.")

    except Exception as e:
        print("❌ Error during parsing:")
        print(str(e))