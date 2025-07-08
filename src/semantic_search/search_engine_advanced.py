# src/rag/query_engine.py

import os
import json
import anthropic
from typing import List
from dataclasses import asdict

from src.rag.retriever import SpreadsheetRetriever
from src.data_ingestion.spreadsheet_parser_advance import ColumnMetadata


class QueryEngine:
    def __init__(self, retriever: SpreadsheetRetriever):
        self.retriever = retriever
        self.api_key = self._load_claude_key()
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _load_claude_key(self) -> str:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        return secrets["claude_api_key"]

    def _build_context(self, columns: List[ColumnMetadata]) -> str:
        context = []
        for col in columns:
            context.append({
            'sheet': col.sheet,
            'header': col.header,
            'data_type': col.data_type,
            'cell_range': ', '.join(col.addresses),  # use stored addresses directly
            'formula': col.first_cell_formula,
            'cross_sheet_refs': col.cross_sheet_refs,
            'sample_values': col.sample_values,
        })
        return json.dumps(context, indent=2)

    def ask(self, query: str) -> str:
        top_columns = self.retriever.retrieve(query, top_k=5)
        top_metadata = [col for col, _ in top_columns]

        context = self._build_context(top_metadata)

        system_prompt = (
            "You are a spreadsheet AI assistant. Given a user query and structured spreadsheet context, "
            "provide a precise and helpful answer using only the context provided.\n\n"
            "Your response must follow this JSON format:\n"
            "[{ concept_name, sheet, header, cell_range, formula, explanation }]"
            "If cell range contains continuous cells then represent like A1:A13 instead of A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13"
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

    engine = QueryEngine(retriever)

    query = input("🔍 Enter query: ")
    response = engine.ask(query)
    print("\n📥 Claude's Response:\n")
    print(response)
