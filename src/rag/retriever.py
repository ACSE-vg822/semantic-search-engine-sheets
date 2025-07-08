# src/rag/retriever.py

import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
from difflib import SequenceMatcher

from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetKnowledgeGraph, ColumnMetadata

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class SpreadsheetRetriever:
    def __init__(self, knowledge_graph: SpreadsheetKnowledgeGraph, use_embeddings: bool = True, debug: bool = False, model: Optional['SentenceTransformer'] = None):
        self.kg = knowledge_graph
        self.entries = self._build_corpus()
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.debug = debug
        if self.use_embeddings:
            # Use provided model or load a new one
            self.model = model if model is not None else SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = self.model.encode([entry["text"] for entry in self.entries], convert_to_tensor=True)

    def _build_corpus(self) -> List[Dict]:
        entries = []
        for sheet_name, sheet in self.kg.sheets.items():
            for col_name, col_meta in sheet.columns.items():
                text = f"{col_meta.header} ({col_meta.data_type}) " + " ".join(str(v) for v in col_meta.sample_values)
                entries.append({
                    "text": text,
                    "sheet": sheet_name,
                    "column": col_meta.header,
                    "metadata": col_meta
                })
        return entries

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[ColumnMetadata, float]]:
        if self.use_embeddings:
            query_emb = self.model.encode(query, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
            scored_entries = sorted(
                zip(self.entries, scores),
                key=lambda x: x[1],
                reverse=True
            )
            results = [(entry["metadata"], float(score)) for entry, score in scored_entries[:top_k]]
        else:
            results = []
            for entry in self.entries:
                sim_score = self._similarity(query, entry["text"])
                if query.lower() in entry["text"]:
                    sim_score += 0.2
                results.append((entry["metadata"], sim_score))
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        if self.debug:
            print("\n[DEBUG] Raw results:")
            for meta, score in results:
                print(f"{meta.sheet} -> {meta.header} : {score:.3f}")

        return results


# 🧪 Test block
if __name__ == "__main__":
    from src.data_ingestion.spreadsheet_parser_advance import SpreadsheetParserAdvanced

    TEST_SPREADSHEET_ID = "1a0coLtHsNNedSu5LZtqh7k3SBkDGG_IeJEHn-ijW9ls"

    parser = SpreadsheetParserAdvanced()
    spreadsheet = parser.parse(TEST_SPREADSHEET_ID)
    kg = parser.build_knowledge_graph(spreadsheet)

    retriever = SpreadsheetRetriever(kg, use_embeddings=True, debug=True)
    query = input("🔍 Enter a query: ")

    results = retriever.retrieve(query)
    for meta, score in results:
        print(f"\n🧠 Match (Score: {score:.2f}):")
        print(json.dumps(asdict(meta), indent=2))
