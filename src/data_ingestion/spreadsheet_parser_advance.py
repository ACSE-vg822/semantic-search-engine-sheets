# spreadsheet_parser_advance.py

import os
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import gspread
from google.oauth2.service_account import Credentials

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
CROSS_SHEET_REGEX = re.compile(r"(?:'([^']+)')|(?:(\w+)![$]?[A-Z]+[$]?\d+)")


@dataclass
class CellInfo:
    row: int
    col: int
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
    description: Optional[str] = None
    cross_sheet_refs: Optional[List[str]] = None



@dataclass
class SheetMetadata:
    name: str
    columns: Dict[str, ColumnMetadata]


@dataclass
class SpreadsheetKnowledgeGraph:
    title: str
    sheets: Dict[str, SheetMetadata]


class SpreadsheetParserAdvanced:
    def __init__(self):
        credentials_dict = self._load_credentials()
        creds = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        self.client = gspread.authorize(creds)

    def _load_credentials(self) -> dict:
        import toml
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if not os.path.exists(secrets_path):
            raise FileNotFoundError("Missing .streamlit/secrets.toml")
        return toml.load(secrets_path)["google_credentials"]

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
                    cell_map[addr] = CellInfo(row=r, col=c, value=val, data_type=data_type, formula=raw_formula)
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
        for sheet_name, cell_map in spreadsheet.cells.items():
            headers = {
                cell.col: str(cell.value).strip()
                for addr, cell in cell_map.items()
                if cell.row == 1 and cell.value
            }

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
                    # Extract sheet names from matches
                    cross_refs = list({m[0] or m[1] for m in matches if (m[0] or m[1])})

                col_meta = ColumnMetadata(
                    header=header,
                    data_type=col_cells[0].data_type if col_cells else "unknown",
                    first_cell_formula=first_formula_cell.formula if first_formula_cell else None,
                    sheet=sheet_name,
                    sample_values=sample_values,
                    cross_sheet_refs=cross_refs or None
                )
                columns[header] = col_meta

            sheets[sheet_name] = SheetMetadata(name=sheet_name, columns=columns)

        return SpreadsheetKnowledgeGraph(title=spreadsheet.title, sheets=sheets)


# 🧪 Test block
if __name__ == "__main__":
    TEST_SPREADSHEET_ID = "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"  # Sales Dashboard

    parser = SpreadsheetParserAdvanced()
    spreadsheet = parser.parse(TEST_SPREADSHEET_ID)
    graph = parser.build_knowledge_graph(spreadsheet)

    print(f"\n📊 Knowledge Graph for Spreadsheet: {graph.title}")
    for sheet_name, sheet_meta in graph.sheets.items():
        print(f"\n--- Sheet: {sheet_name} ---")
        for col_name, col_meta in sheet_meta.columns.items():
            print(json.dumps(asdict(col_meta), indent=2))
            #break  # Only show one column
        break  # Only show one sheet
