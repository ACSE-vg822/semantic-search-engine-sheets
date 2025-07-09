# spreadsheet_parser_advance.py

import os
import json
import logging
import toml
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import gspread
import streamlit as st
from google.oauth2 import service_account

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
CROSS_SHEET_REGEX = re.compile(r"(?:'([^']+)')|(?:(\w+)![$]?[A-Z]+[$]?\d+)")


@dataclass
class CellInfo:
    row: int
    col: int
    address: str  # Add this field
    value: Union[str, float, int, None]
    data_type: str
    formula: Optional[str]


@dataclass
class SpreadsheetData:
    title: str
    sheet_names: List[str]
    cells: Dict[str, Dict[str, CellInfo]]  # sheet_name -> cell_address -> CellInfo


@dataclass
class ColumnMetadata:
    header: str
    data_type: str
    first_cell_formula: Optional[str]
    sheet: str
    sample_values: List[Union[str, float, int]]
    addresses: str  # Changed to str for range format (e.g., "A2:A13")
    cross_sheet_refs: Optional[List[str]] = None

@dataclass
class RowMetadata:
    concept: str                     # Value in first column (e.g. A4 = "Gross Profit")
    sheet: str
    row_number: int
    data_type: str
    sample_values: List[Union[str, float, int]]
    formulae: List[str]
    cell_addresses: List[str]
    col_headers: List[str]
    cross_sheet_refs: Optional[List[str]] = None

@dataclass
class SheetMetadata:
    name: str
    columns: Dict[str, ColumnMetadata]


@dataclass
class SpreadsheetKnowledgeGraph:
    title: str
    sheets: Dict[str, SheetMetadata]
    rows: Optional[Dict[str, List[RowMetadata]]] = None


class SpreadsheetParserAdvanced:
    def __init__(self):
        self.client = self._setup_google_sheets_auth()

    def _setup_google_sheets_auth(self):
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        return gspread.authorize(credentials)

    def parse(self, spreadsheet_id: str) -> SpreadsheetData:
        spreadsheet = self.client.open_by_key(spreadsheet_id)
        sheet_names = [s.title for s in spreadsheet.worksheets()]
        all_cells = {}

        for sheet in spreadsheet.worksheets():
            cell_map = {}
            data = sheet.get_all_values()
            formulas = sheet.get_all_values(value_render_option='FORMULA')
            
            for r, (row, formula_row) in enumerate(zip(data, formulas), start=1):
                for c, (val, formula) in enumerate(zip(row, formula_row), start=1):
                    addr = self._cell_address(r, c)
                    formula_str = str(formula) if formula is not None else ""
                    raw_formula = formula_str if formula_str.startswith("=") else None
                    data_type = self._infer_type(val, raw_formula)
                    addr = self._cell_address(r, c)
                    cell_map[addr] = CellInfo(
                        row=r,
                        col=c,
                        address=addr,  # pass actual cell address here
                        value=val,
                        data_type=data_type,
                        formula=raw_formula
                    )
            all_cells[sheet.title] = cell_map

        return SpreadsheetData(title=spreadsheet.title, sheet_names=sheet_names, cells=all_cells)

    def _cell_address(self, row: int, col: int) -> str:
        letters = ""
        while col > 0:
            col, rem = divmod(col - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row}"

    def _infer_type(self, val: str, formula: Optional[str]) -> str:
        if formula:
            return "formula"
        if val.replace(".", "", 1).isdigit():
            return "number"
        return "text"

    def _normalize_formula(formula: str) -> str:
        import re
        return re.sub(r'\d+', 'n', formula)

    def build_knowledge_graph(self, spreadsheet: SpreadsheetData) -> SpreadsheetKnowledgeGraph:
        sheets = {}
        row_level_data = {}

        for sheet_name, cell_map in spreadsheet.cells.items():
            headers = {
                cell.col: str(cell.value).strip()
                for addr, cell in cell_map.items()
                if cell.row == 1 and cell.value
            }

            # === COLUMN METADATA ===
            columns = {}
            for col_idx, header in headers.items():
                col_cells = [
                    cell for cell in cell_map.values()
                    if cell.col == col_idx and cell.row > 1
                ]
                sample_values = [cell.value for cell in col_cells[:5] if cell.value not in [None, ""]]
                first_formula_cell = next((c for c in col_cells if c.formula), None)

                cross_refs = []
                if first_formula_cell and first_formula_cell.formula:
                    matches = CROSS_SHEET_REGEX.findall(first_formula_cell.formula)
                    cross_refs = list({m[0] or m[1] for m in matches if (m[0] or m[1])})

                selected_cells = col_cells[:13]
                addresses = f"{selected_cells[0].address}:{selected_cells[-1].address}" if selected_cells else ""

                col_meta = ColumnMetadata(
                    header=header,
                    data_type=col_cells[0].data_type if col_cells else "unknown",
                    first_cell_formula=first_formula_cell.formula if first_formula_cell else None,
                    sheet=sheet_name,
                    sample_values=sample_values,
                    cross_sheet_refs=cross_refs or None,
                    addresses=addresses
                )
                columns[header] = col_meta

            sheets[sheet_name] = SheetMetadata(name=sheet_name, columns=columns)

            # === ROW METADATA (for row-major use cases) ===
            sheet_rows = {}
            for row_idx in range(2, max(c.row for c in cell_map.values()) + 1):
                row_cells = [
                    cell for cell in cell_map.values()
                    if cell.row == row_idx
                ]
                if not row_cells:
                    continue

                row_values = [cell.value for cell in row_cells if cell.value not in [None, ""]]
                row_formulae = [cell.formula for cell in row_cells if cell.formula]
                row_headers = [headers.get(cell.col, "") for cell in row_cells]

                # Extract cross-sheet references from any cell
                all_formulas = " ".join(f for f in row_formulae if f)
                matches = CROSS_SHEET_REGEX.findall(all_formulas)
                cross_refs = list({m[0] or m[1] for m in matches if (m[0] or m[1])})

                concept = str(row_cells[0].value) if row_cells else f"Row {row_idx}"

                # Convert address list to range (e.g., "A6:F6")
                row_cells_sorted = sorted(row_cells, key=lambda c: c.col)
                if row_cells_sorted:
                    start_address = row_cells_sorted[0].address
                    end_address = row_cells_sorted[-1].address
                    address_range = f"{start_address}:{end_address}"
                else:
                    address_range = ""

                row_meta = RowMetadata(
                    concept=concept,
                    sheet=sheet_name,
                    row_number=row_idx,
                    data_type=row_cells[1].data_type if len(row_cells) > 1 else "unknown",
                    sample_values=row_values[:5],
                    formulae=row_formulae,
                    cell_addresses=address_range,
                    col_headers=row_headers,
                    cross_sheet_refs=cross_refs or None
                )

                sheet_rows[row_idx] = row_meta

            row_level_data[sheet_name] = list(sheet_rows.values())

        return SpreadsheetKnowledgeGraph(title=spreadsheet.title, sheets=sheets, rows=row_level_data)




# 🧪 Test block
if __name__ == "__main__":
    TEST_SPREADSHEET_ID = "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"  # Sales Dashboard

    parser = SpreadsheetParserAdvanced()
    spreadsheet = parser.parse(TEST_SPREADSHEET_ID)
    graph = parser.build_knowledge_graph(spreadsheet)

    print(f"\n📊 Knowledge Graph for Spreadsheet: {graph.title}")

    # Show column-wise metadata
    print("\n🧱 COLUMN-WISE METADATA:")
    for sheet_name, sheet_meta in graph.sheets.items():
        print(f"\n--- Sheet: {sheet_name} ---")
        for col_name, col_meta in sheet_meta.columns.items():
            print(json.dumps(asdict(col_meta), indent=2))
        break  # Only show one sheet

    # Show row-wise metadata
    if hasattr(graph, "rows"):  # Ensure compatibility
        print("\n📋 ROW-WISE METADATA:")
        for sheet_name, row_list in graph.rows.items():
            print(f"\n--- Sheet: {sheet_name} ---")
            for row_meta in row_list[:5]:  # Only print first 5 rows
                print(json.dumps(asdict(row_meta), indent=2))
            break  # Only show one sheet
