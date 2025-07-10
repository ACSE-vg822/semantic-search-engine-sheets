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
  - Handles cross-sheet reference detection

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
- **Supported Operations**: sum, average, count, min, max, divide, multiply, subtract, add, percentage
- **Query Parsing**: Uses Claude LLM to parse natural language into structured calculation requests

#### **User Interface** (`streamlit_app.py`)

- **Streamlit-based web interface**
- **Features**:
  - Spreadsheet ID input with example IDs
  - Real-time query processing
  - Expandable result cards with detailed metadata
  - Caching for improved performance

## 3. Technology Stack

### 3.1 Core Dependencies

```python
# AI/ML Stack
anthropic              # Claude LLM integration
sentence-transformers  # Semantic embeddings
langgraph             # Workflow orchestration
langchain             # LLM framework
langchain-anthropic   # Claude integration

# Data Processing
pandas                # Data manipulation
numpy                 # Numerical computing
gspread              # Google Sheets API
google-auth          # Authentication

# Web Interface
streamlit            # UI framework
plotly               # Data visualization

# Utilities
pydantic             # Data validation
networkx             # Graph operations
```

### 3.2 External Services

- **Google Sheets API**: Source data access
- **Anthropic Claude**: LLM for query classification and parsing
- **Sentence Transformers**: Semantic embeddings via Hugging Face models

## 4. Core Workflows

### 4.1 Search Workflow

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

### 4.2 Calculation Workflow

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
4. Return tuples: `(metadata_object, similarity_score, entry_type)`

### 5.3 LangGraph Orchestration

**State Management**: Typed dictionaries track workflow state
**Node Functions**:

- **Classifier Node**: Claude LLM determines query intent
- **Search Node**: RAG retrieval + LLM filtering
- **Calculate Node**: Calculation engine integration

**Routing Logic**: Conditional edges based on classification results

### 5.4 Calculation Engine

**Query Parsing**: Claude LLM extracts:

```python
@dataclass
class CalculationRequest:
    operation: Literal["sum", "average", "count", ...]
    target_concepts: List[str]  # e.g., ["actual revenue", "net profit"]
    filters: Optional[Dict[str, str]]  # e.g., {"sheet": "Q1 Data"}
```

**Data Identification**: RAG-based search for relevant numerical data
**Value Extraction**: Fresh spreadsheet data fetch for current values
**Calculation Execution**: Mathematical operations on extracted data

## 6. Key Features & Capabilities

### 6.1 Semantic Understanding

**Business Concept Recognition**:

- Revenue ≈ Sales ≈ Income
- Profit ≈ Earnings ≈ Net Income
- Efficiency ≈ Productivity ≈ Performance

**Context Interpretation**:

- Distinguishes "Marketing Spend" (cost) from "Marketing ROI" (efficiency)
- Recognizes formula semantics: `=B5/B6` in "Margin %" column = margin calculation

### 6.2 Natural Language Processing

**Query Types Supported**:

- **Conceptual**: "Find all profitability metrics"
- **Functional**: "Show percentage calculations"
- **Comparative**: "Budget vs actual analysis"
- **Computational**: "Calculate total revenue"

### 6.3 Multi-Sheet Intelligence

**Cross-Sheet Understanding**:

- Reference detection using regex: `(?:'([^']+)')|(?:(\w+)![$]?[A-Z]+[$]?\d+)`
- Relationship tracking between related sheets
- Context-aware search across worksheet boundaries

### 6.4 Result Intelligence

**Ranking Factors**:

- Semantic relevance score
- Business context importance
- Formula complexity analysis
- Data recency considerations

**Output Format**:

- Contextual explanations
- Relevance justifications
- Source data transparency
- Business concept mapping

## 7. Performance & Scalability

### 7.1 Optimization Strategies

**Caching Architecture**:

- Knowledge graph caching in Streamlit session state
- Retriever instance reuse across queries
- Embedding precomputation and storage

**Lazy Loading**:

- Spreadsheet parser initialization on-demand
- Fresh data fetching only for calculations

### 7.2 Current Limitations

**Scale Constraints**:

- In-memory embedding storage
- Single-spreadsheet focus
- Real-time API calls for calculations

**Processing Bottlenecks**:

- Google Sheets API rate limits
- LLM inference latency
- Large spreadsheet parsing time

## 8. Security & Authentication

### 8.1 Google Sheets Access

**Service Account Authentication**:

```toml
[google_credentials]
type = "service_account"
project_id = "your-project-id"
private_key = "-----BEGIN PRIVATE KEY-----\n..."
client_email = "service-account@project.iam.gserviceaccount.com"
```

**Permission Model**: Read-only access to shared spreadsheets

### 8.2 API Key Management

**Claude API**: Secured via Streamlit secrets
**Access Control**: Application-level authentication required

## 9. Testing & Validation

### 9.1 Test Data Sources

**Provided Test Spreadsheets**:

- Financial Model: Complex financial calculations and projections
- Sales Dashboard: Sales metrics and performance data

### 9.2 Evaluation Criteria

**Search Quality**:

- Semantic relevance of retrieved results
- Business context accuracy
- Cross-sheet relationship detection

**Calculation Accuracy**:

- Mathematical operation correctness
- Data source identification precision
- Result explanation clarity

## 10. Future Enhancements

### 10.1 Planned Improvements

**Real-Time Updates**: Live spreadsheet change monitoring
**Enhanced Multi-Sheet**: Advanced cross-sheet analytics
**Performance Optimization**: Distributed caching and processing
**Advanced Calculations**: Complex formula interpretation

### 10.2 Scalability Roadmap

**Enterprise Features**:

- Multi-spreadsheet corpus management
- User access control and permissions
- Advanced analytics and reporting
- Integration with business intelligence tools

## 11. Conclusion

This semantic search engine represents a significant advancement in spreadsheet interaction, bridging the gap between how users think about data and how they can access it. By combining state-of-the-art NLP techniques with domain-specific knowledge representation, the system enables intuitive, natural language-based exploration of complex spreadsheet data.

The modular architecture supports extensibility while maintaining performance, and the use of established frameworks (LangGraph, LangChain, Streamlit) ensures maintainability and developer productivity. The system successfully addresses the core challenge of semantic search in business documents while providing a foundation for advanced analytical capabilities.
