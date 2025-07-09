# Semantic Search Engine for Spreadsheets - Comprehensive Design Document

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)

---

## Project Overview

### Problem Statement

Traditional spreadsheet search tools only support structural queries (exact text matches, cell references) but users think semantically about business concepts. This project bridges the gap between conceptual thinking and spreadsheet structure.

### Solution

A semantic search engine that understands business concepts and allows natural language queries like:

- "Find all profitability metrics"
- "Show me cost calculations"
- "Where are my growth rates?"
- "Find efficiency ratios"

### Key Innovation

Combines **Knowledge Graph representation**, **RAG (Retrieval-Augmented Generation)**, and **LLM intelligence** to understand spreadsheet content conceptually rather than structurally.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Query Engine    │───▶│   Claude AI     │
│   (app_new.py)  │    │(search_engine.py)│    │  (Anthropic)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│ Knowledge Graph │    │   RAG Retriever  │
│   (Cached)      │    │ (retriever.py)   │
└─────────────────┘    └──────────────────┘
         ▲                        ▲
         │                        │
┌─────────────────┐    ┌──────────────────┐
│Data Ingestion   │    │Sentence Embeddings│
│(parser.py)      │    │   (Transformers) │
└─────────────────┘    └──────────────────┘
         ▲
         │
┌─────────────────┐
│  Google Sheets  │
│      API        │
└─────────────────┘
```

### Technology Stack

- **Frontend**: Streamlit (Interactive web UI)
- **Backend**: Python 3.x
- **ML/AI**:
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - Claude 3.7 Sonnet (Anthropic)
- **Data Sources**: Google Sheets API
- **Authentication**: Google Service Account
- **Dependencies**: See `requirements.txt`

---

## Core Components

### 1. Data Ingestion Layer (`src/data_ingestion/spreadsheet_parser_advance.py`)

#### Purpose

Extracts and structures spreadsheet data into semantic knowledge graphs.

#### Key Classes

```python
@dataclass
class CellInfo:
    row: int
    col: int
    address: str        # e.g., "A5"
    value: Union[str, float, int, None]
    data_type: str      # "text", "number", "formula"
    formula: Optional[str]

@dataclass
class ColumnMetadata:
    header: str                    # Column header name
    data_type: str                # Data type classification
    first_cell_formula: str       # First formula found
    sheet: str                    # Parent sheet name
    sample_values: List           # First 5 values for context
    addresses: str               # Cell range (e.g., "A2:A13")
    cross_sheet_refs: List[str]  # Referenced sheets

@dataclass
class RowMetadata:
    first_cell_value: str        # Business concept (first column value)
    sheet: str                   # Parent sheet
    row_number: int             # Row position
    data_type: str              # Dominant data type
    sample_values: List         # Row values
    formulae: List[str]         # All formulas in row
    cell_addresses: str         # Row range
    col_headers: List[str]      # Column headers
    cross_sheet_refs: List[str] # Cross-references

@dataclass
class SpreadsheetKnowledgeGraph:
    title: str                                    # Spreadsheet title
    sheets: Dict[str, SheetMetadata]             # Sheet-wise column data
    rows: Dict[str, List[RowMetadata]]           # Sheet-wise row data
```

#### Core Functionality

1. **Google Sheets Authentication**: Service account-based secure access
2. **Data Extraction**: Retrieves values AND formulas for complete context
3. **Type Inference**: Automatically classifies cells as text/number/formula
4. **Dual Representation**: Creates both column-centric and row-centric views

### 2. RAG Retrieval System (`src/rag/retriever.py`)

#### Purpose

Converts knowledge graphs into searchable semantic representations using vector embeddings.

#### Embedding Strategy

```python
class SpreadsheetRetriever:
    def __init__(self, knowledge_graph, use_embeddings=True, model=None):
        self.kg = knowledge_graph
        self.entries = self._build_corpus()  # Create searchable text
        if use_embeddings:
            self.model = model or SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = self.model.encode([entry["text"] for entry in self.entries])
```

#### Corpus Building

- **Column Entries**: `"Column: {header} ({data_type}) {sample_values}"`
- **Row Entries**: `"Row concept: {concept} ({data_type}) {sample_values}"`

#### Similarity Calculation

1. **Primary**: Cosine similarity with sentence transformers
2. **Fallback**: SequenceMatcher-based text similarity
3. **Boost**: Keyword matching bonus for exact term presence

#### Retrieval Process

```python
def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Metadata, float, str]]:
    if self.use_embeddings:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
        # Return top-k results with metadata, score, and type
    else:
        # Fallback to text similarity
```

### 3. Semantic Search Engine (`src/semantic_search/search_engine_advanced.py`)

#### Purpose

Orchestrates the end-to-end query processing using Claude AI for business intelligence.

#### Query Processing Pipeline

1. **Retrieval**: Get top 10 relevant columns/rows from RAG system
2. **Data Enhancement**: Fetch real-time values and calculate statistics
3. **Context Building**: Create structured JSON context for Claude
4. **LLM Processing**: Send enhanced context to Claude with business-focused prompts
5. **Response Parsing**: Return structured JSON results

#### Statistical Enhancement

```python
def _calculate_numerical_stats(self, values):
    return {
        'min': min(numeric_values),
        'max': max(numeric_values),
        'mean': sum(numeric_values) / len(numeric_values),
        'count': len(numeric_values),
        'total_cells': len(all_values)
    }
```

#### Claude AI Integration

- **Model**: Claude 3.7 Sonnet (latest high-performance model)
- **Context**: Up to 8,192 tokens for comprehensive context
- **Temperature**: 0.3 (balanced creativity vs accuracy)
- **System Prompt**: Sophisticated business intelligence prompting

#### Business Intelligence System Prompt

The system prompt includes:

1. **Concept Recognition**: Financial, operational, temporal, performance metrics
2. **Synonym Handling**: Revenue=sales=income, profit=earnings, etc.
3. **Context Interpretation**: Understanding formula meaning based on column context
4. **Formula Semantics**: SUM in "Total Sales" = revenue aggregation
5. **Output Formatting**: Structured JSON with business explanations

### 4. User Interface (`app_new.py`)

#### Streamlit Architecture

- **Session State Management**: Persistent caching of models and knowledge graphs
- **Multi-Input Support**: Predefined test sheets OR custom Google Sheets
- **Progressive Loading**: Staged loading with progress indicators
- **Error Handling**: Graceful degradation and user feedback

#### Caching Strategy

```python
# Session state caching for performance
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'sentence_transformer_model' not in st.session_state:
    st.session_state.sentence_transformer_model = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
```

#### User Experience Features

1. **Smart Spreadsheet Selection**: Dropdown OR custom URL input
2. **Sample Queries**: One-click example questions
3. **Real-time Processing**: Progress indicators for each stage
4. **Rich Results Display**: Expandable cards with formulas and explanations
5. **Formula Normalization**: Readable formula display with subscript notation
