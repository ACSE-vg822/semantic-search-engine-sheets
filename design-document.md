# Design Document: Semantic Search Engine for Spreadsheets

## 1. Executive Summary

### Project Overview

The Semantic Search Engine for Spreadsheets is an AI-powered system that enables users to query Google Sheets data using natural language instead of exact text matches. The system understands business concepts, interprets context, and provides intelligent search and calculation capabilities across spreadsheet data.

### Key Value Proposition

- **Semantic Understanding**: Recognizes that "Q1 Revenue", "First Quarter Sales", and "Jan-Mar Income" refer to similar concepts
- **Natural Language Queries**: Processes queries like "show efficiency metrics" or "find budget vs actual comparisons"
- **Intelligent Calculations**: Performs calculations based on natural language requests like "calculate total revenue"
- **Cross-Sheet Intelligence**: Understands relationships and references across multiple spreadsheet tabs

### Core Capabilities

1. **Conceptual Search**: Find data by meaning rather than exact text
2. **Natural Language Calculations**: Execute mathematical operations through conversational queries
3. **Business Context Awareness**: Understand domain-specific terminology and relationships
4. **Multi-Sheet Analysis**: Analyze data across different worksheets within a spreadsheet

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀──▶│  LangGraph       │◀───│   Google        │
│                 │    │  Search Engine   │    │   Sheets API    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▲                        │
                                │                        │
                       ┌──────────────────┐              │
                       │   RAG Retriever  │              │
                       │  + Embeddings    │              │
                       └──────────────────┘              │
                                ▲                        │
                                │                        ▼
                       ┌──────────────────┐◀─────────────┘
                       │  Knowledge Graph │
                       │   (Structured    │
                       │   Spreadsheet    │
                       │     Metadata)    │
                       └──────────────────┘
```

### 2.2 Core Components

#### **Data Ingestion Layer** (`src/data_ingestion/`)

- **SpreadsheetParserAdvanced**: Primary parser for Google Sheets data
- **Responsibilities**:
  - Authenticates with Google Sheets API using service account credentials
  - Parses spreadsheet content including values, formulas, and metadata
  - Builds structured knowledge graph from raw spreadsheet data

#### **Knowledge Representation** (Data Models)

```python
@dataclass
class CellInfo:
    row: int
    col: int
    address: str
    value: Union[str, float, int, None]
    data_type: str
    formula: Optional[str]

@dataclass
class ColumnMetadata:
    header: str
    data_type: str
    first_cell_formula: Optional[str]
    sheet: str
    sample_values: List[Union[str, float, int]]
    addresses: str
    cross_sheet_refs: Optional[List[str]]

@dataclass
class RowMetadata:
    first_cell_value: str
    sheet: str
    row_number: int
    data_type: str
    sample_values: List[Union[str, float, int]]
    formulae: List[str]
    addresses: str
    col_headers: List[str]
    cross_sheet_refs: Optional[List[str]]
```

#### **RAG Retrieval System** (`src/rag/`)

- **SpreadsheetRetriever**: Semantic similarity-based retrieval
- **Technology Stack**:
  - Sentence Transformers (`all-MiniLM-L6-v2` model)
  - PyTorch tensors for embedding storage
  - Cosine similarity for relevance scoring
- **Corpus Construction**:
  - Column-based entries: "Column: {header} ({data_type}) {sample_values}"
  - Row-based entries: "Row concept: {first_cell_value} ({data_type}) {sample_values}"

#### **Search Engine Orchestration** (`src/semantic_search/`)

- **LangGraphSearchEngine**: Main orchestration using LangGraph workflow
- **Workflow States**:
  ```python
  class SearchState(TypedDict):
      user_query: str
      query_type: Optional[Literal["search", "calculate"]]
      rag_results: Optional[List[...]]
      filtered_results: Optional[List[Dict]]
      explanation: Optional[str]
      final_response: Optional[str]
  ```

#### **Calculation Engine** (`src/semantic_search/`)

- **CalculationEngine**: Handles computational queries
- **Supported Operations**: sum, average, count, min, max
- **Query Parsing**: Uses Claude LLM to parse natural language into structured calculation requests

#### **User Interface** (`streamlit_app.py`)

- **Streamlit-based web interface**
- **Features**:
  - Spreadsheet ID input with example IDs
  - Real-time query processing
  - Expandable result cards with detailed metadata
  - Caching for improved performance

## 3. Core Workflows

### 3.1 Search Workflow

```
User Query Input
      │
      ▼
┌─────────────────┐
│   Classifier    │ ──▶ Claude LLM determines: "search" vs "calculate"
│     Node        │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Search Node   │ ──▶ 1. RAG Retrieval (top-k results)
│                 │     2. LLM Filtering & Explanation
│                 │     3. Result Formatting
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Final Response │
└─────────────────┘
```

### 3.2 Calculation Workflow

```
User Query Input
      │
      ▼
┌─────────────────┐
│   Classifier    │ ──▶ Identifies as "calculate"
│     Node        │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Calculate Node  │ ──▶ 1. Parse calculation request
│                 │     2. Identify relevant data via RAG
│                 │     3. Extract numeric values
│                 │     4. Perform calculation
│                 │     5. Format results
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Final Response │
└─────────────────┘
```

## 5. Detailed Component Analysis

### 5.1 Data Ingestion & Knowledge Graph Construction

**SpreadsheetParserAdvanced** transforms raw Google Sheets data into a structured knowledge graph:

**Input**: Google Spreadsheet ID
**Process**:

1. **Authentication**: Service account credentials via Streamlit secrets
2. **Data Extraction**:
   - Cell values and formulas from all worksheets
   - Cross-sheet reference detection using regex patterns
3. **Metadata Construction**:
   - Column-level metadata with headers, data types, sample values
   - Row-level metadata with first cell values and formula analysis
4. **Knowledge Graph**: Structured representation enabling semantic search

**Output**: `SpreadsheetKnowledgeGraph` with hierarchical metadata

### 5.2 Semantic Retrieval System

**SpreadsheetRetriever** implements RAG (Retrieval-Augmented Generation):

**Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
**Corpus Construction**:

```python
# Column entries
"Column: Revenue (number) 150000 200000 175000"

# Row entries
"Row concept: Gross Profit (number) 45000 60000 52500"
```

**Retrieval Process**:

1. Query embedding generation
2. Cosine similarity computation against corpus
3. Top-k result selection with relevance scores

## 5. Future Enhancements

### 5.1 Planned Improvements

**Enhanced Multi-Node Calculation Architecture**: The current single-node calculation flow limits our ability to handle complex, multi-step calculations that require validation, error correction, and iterative refinement. A sophisticated calculation system requires a multi-node LangGraph architecture with the following components:

#### **Calculation Workflow 2.0: Multi-Node Architecture**

```
User Calculation Query
         │
         ▼
┌─────────────────┐
│   Planner Node  │ ──▶ 1. Parse complex calculation request
│                 │     2. Break down into sub-calculations
│                 │     3. Identify data dependencies
│                 │     4. Create execution plan with steps
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Validator Node  │ ──▶ 1. Validate plan feasibility
│                 │     2. Check data availability
│                 │     3. Verify mathematical logic
│                 │     4. Flag potential issues
└─────────────────┘
         │
   ┌─────┴─────┐
   │ Valid?    │
   └─────┬─────┘
      No │              Yes
         │              │
         ▼              ▼
┌─────────────────┐  ┌─────────────────┐
│ Replanner Node  │  │ Execution Node  │
│                 │  │                 │
│ 1. Identify     │  │ 1. Execute plan │
│    issues       │  │    step-by-step │
│ 2. Adjust plan  │  │ 2. Handle data  │
│ 3. Find alt.    │  │    extraction   │
│    data sources │  │ 3. Perform      │
│ 4. Simplify     │  │    calculations │
│    if needed    │  │ 4. Track inter- │
└─────────┬───────┘  │    mediate      │
          │          │    results      │
          ▲          └─────────────────┘
          │                   │
    (re-validate)             │
          │                   │
                              ▼
                            ┌─────────────────┐
                            │ Result Formatter│ ──▶ 1. Format final results
                            │     Node        │     2. Create explanations
                            │                 │     3. Show calculation steps
                            │                 │     4. Highlight data sources
                            └─────────────────┘
```

#### **Node Responsibilities:**

**Planner Node:**

- Decomposes complex queries into sequential calculation steps
- Maps business concepts to specific data sources
- Creates dependency graphs for multi-step calculations
- Handles queries like "Calculate profit margin trend over 3 quarters"

**Validator Node:**

- Verifies mathematical validity of the plan
- Checks data availability and completeness
- Validates business logic (e.g., ratios should use compatible units)
- Ensures calculations are feasible with available data

**Replanner Node:**

- Triggered when validation fails
- Suggests alternative data sources or calculation methods
- Simplifies complex requests when data is insufficient
- Provides fallback strategies (e.g., approximate calculations)

**Execution Node:**

- Executes validated plans step-by-step
- Maintains state between calculation steps
- Handles intermediate result storage
- Provides detailed audit trail of calculations

#### **Advanced Calculation Capabilities:**

1. **Multi-Step Calculations**: "Calculate ROI for each product line, then rank by performance"
2. **Cross-Sheet Analysis**: "Compare Q1 actuals vs budget across all departments"
3. **Trend Analysis**: "Show revenue growth rate for the last 6 months"
4. **Conditional Logic**: "Sum sales where region is 'North' and quarter is 'Q4'"
5. **Formula Recreation**: "Rebuild the profit calculation used in the original spreadsheet"

#### **Error Handling & Recovery:**

- **Data Validation**: Check for missing values, data type mismatches
- **Calculation Verification**: Validate intermediate results for reasonableness
- **Alternative Strategies**: When primary calculation fails, suggest approximations
- **User Feedback**: Explain why certain calculations cannot be performed

#### **State Management:**

```python
class CalculationState(TypedDict):
    user_query: str
    calculation_plan: Optional[List[CalculationStep]]
    validation_results: Optional[ValidationReport]
    execution_status: Optional[ExecutionProgress]
    intermediate_results: Optional[Dict[str, float]]
    final_result: Optional[CalculationResult]
    error_state: Optional[ErrorContext]
    replanning_attempts: int
```

This architecture enables handling complex business calculations that require multiple data sources, intermediate calculations, and sophisticated error handling - far beyond the current single-step calculation approach.
