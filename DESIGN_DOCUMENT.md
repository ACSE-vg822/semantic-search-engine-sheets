# Semantic Search Engine for Spreadsheets - Comprehensive Design Document

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow & Processing Pipeline](#data-flow--processing-pipeline)
5. [Technical Implementation](#technical-implementation)
6. [User Interface & Experience](#user-interface--experience)
7. [Performance & Optimization](#performance--optimization)
8. [API Integration & Security](#api-integration--security)
9. [Challenges & Solutions](#challenges--solutions)
10. [Future Enhancements](#future-enhancements)

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
    concept: str                 # Business concept (first column value)
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
4. **Cross-Sheet Analysis**: Identifies dependencies between sheets using regex
5. **Dual Representation**: Creates both column-centric and row-centric views

#### Cross-Sheet Reference Detection

```python
CROSS_SHEET_REGEX = re.compile(r"(?:'([^']+)')|(?:(\w+)![$]?[A-Z]+[$]?\d+)")
```

Detects patterns like `'Sheet Name'!A1` or `SheetName!A1` in formulas.

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

---

## Data Flow & Processing Pipeline

### Phase 1: Data Ingestion & Knowledge Graph Construction

```
Google Sheets → Authentication → Raw Data Extraction → Type Inference →
Cross-Sheet Analysis → Knowledge Graph Creation → Session Cache
```

**Detailed Steps:**

1. Authenticate via Google Service Account
2. Extract all cell values AND formulas simultaneously
3. Classify each cell as text/number/formula
4. Identify cross-sheet references in formulas
5. Build dual representation (column & row metadata)
6. Cache knowledge graph in Streamlit session

### Phase 2: Semantic Embedding & Retrieval Setup

```
Knowledge Graph → Corpus Building → Sentence Transformer Loading →
Embedding Generation → Retriever Creation → Session Cache
```

**Detailed Steps:**

1. Convert metadata to searchable text corpus
2. Load `all-MiniLM-L6-v2` model (cached after first load)
3. Generate embeddings for all corpus entries
4. Create retriever with precomputed embeddings
5. Cache retriever for instant query processing

### Phase 3: Query Processing & Enhancement

```
User Query → RAG Retrieval → Real-time Data Fetching → Statistical Analysis →
Context Building → Claude API Call → Response Processing → Results Display
```

**Detailed Steps:**

1. Retrieve top 10 semantically similar entries
2. Separate into columns (top 3) and rows (top 5)
3. Fetch current data from Google Sheets for statistics
4. Calculate min/max/mean for numerical columns
5. Build comprehensive JSON context
6. Send to Claude with business intelligence prompt
7. Parse JSON response and display results

---

## Technical Implementation

### Authentication & Security

```python
# Google Sheets Authentication
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
)

# Claude API Authentication
self.client = anthropic.Anthropic(api_key=st.secrets["claude_api_key"])
```

### Error Handling & Resilience

1. **Graceful Degradation**: Falls back to text similarity if embeddings fail
2. **API Error Handling**: Comprehensive try/catch with user-friendly messages
3. **Input Validation**: URL/ID extraction and validation
4. **Session Recovery**: Smart cache invalidation and rebuilding

### Performance Optimizations

1. **Model Caching**: One-time sentence transformer loading
2. **Knowledge Graph Caching**: Avoid rebuilding for same spreadsheet
3. **Embedding Precomputation**: Generate embeddings once, use repeatedly
4. **Batch Processing**: Single API calls for data extraction
5. **Lazy Loading**: Components loaded only when needed

### Data Structures & Memory Management

- **Dataclasses**: Type-safe, memory-efficient data representation
- **Session State**: Streamlit's built-in caching mechanism
- **Sparse Storage**: Only store non-empty cells and relevant metadata
- **Garbage Collection**: Clear caches when switching spreadsheets

---

## User Interface & Experience

### Interface Design Philosophy

- **Progressive Disclosure**: Show complexity only when needed
- **Immediate Feedback**: Real-time progress indicators
- **Error Prevention**: Clear validation and guidance
- **Accessibility**: Structured layouts and clear typography

### User Journey

1. **Spreadsheet Selection**: Choose predefined or add custom
2. **Knowledge Graph Building**: Automatic with progress indication
3. **Query Input**: Natural language with sample suggestions
4. **Results Display**: Rich, contextual results with explanations
5. **Iteration**: Easy query refinement and exploration

### Result Presentation

```python
def display_search_results(parsed_results):
    for i, res in enumerate(parsed_results, 1):
        title = f"📌 Result {i}: {header} | {sheet_name}"
        with st.expander(title):
            st.markdown(f"**📋 Sheet:** `{sheet_name}`")
            st.markdown(f"**📊 Header:** `{header}`")
            st.markdown(f"**🔢 Formula:** {normalized_formula}")
            st.markdown(f"**📝 Explanation:** {explanation}")
```

### Sample Query Interface

Provides one-click access to common business queries:

- 💰 Find all revenue calculations
- 💸 Show me cost-related formulas
- 📊 Where are my margin analyses?
- 📈 What percentage calculations do I have?

---

## Performance & Optimization

### Caching Strategy

1. **Model Caching**: 384MB sentence transformer loaded once
2. **Knowledge Graph Caching**: Avoid re-parsing same spreadsheet
3. **Embedding Caching**: Precompute and store all embeddings
4. **Session Persistence**: Maintain state across queries

### Performance Metrics

- **Initial Load**: ~10-15 seconds (model + knowledge graph + embeddings)
- **Subsequent Queries**: ~2-3 seconds (cached retrieval + Claude API)
- **Memory Usage**: ~500MB (model + embeddings + knowledge graph)
- **Spreadsheet Size**: Tested up to 1000+ cells across multiple sheets

### Optimization Techniques

```python
# Batch embedding generation
self.embeddings = self.model.encode(
    [entry["text"] for entry in self.entries],
    convert_to_tensor=True
)

# Efficient similarity calculation
scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
```

### Scalability Considerations

- **Memory**: Linear growth with spreadsheet size
- **Compute**: One-time embedding cost, fast retrieval
- **API Limits**: Rate limiting for Google Sheets and Claude APIs
- **Concurrent Users**: Session isolation via Streamlit

---

## API Integration & Security

### Google Sheets API

- **Authentication**: Service Account with read-only permissions
- **Scope**: `https://www.googleapis.com/auth/spreadsheets.readonly`
- **Rate Limits**: Built-in `gspread` throttling
- **Error Handling**: Graceful handling of permission/access errors

### Claude API (Anthropic)

- **Model**: Claude 3.7 Sonnet (latest version)
- **Authentication**: API key via Streamlit secrets
- **Context Limits**: 8,192 token responses
- **Rate Limits**: Built-in client throttling
- **Error Handling**: Retry logic and fallback responses

### Security Measures

1. **Secrets Management**: All credentials via Streamlit secrets
2. **Read-Only Access**: No write permissions to spreadsheets
3. **Input Validation**: URL/ID sanitization and validation
4. **Error Sanitization**: No sensitive data in error messages
5. **Session Isolation**: User data isolated in session state

### Deployment Configuration

```toml
# .streamlit/secrets.toml
claude_api_key = "your_claude_api_key"

[google_credentials]
type = "service_account"
project_id = "your_project_id"
private_key_id = "your_private_key_id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your_service_account@project.iam.gserviceaccount.com"
client_id = "your_client_id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

---

## Challenges & Solutions

### Challenge 1: Business Concept Understanding

**Problem**: Traditional keyword search can't understand that "revenue", "sales", and "income" refer to the same concept.

**Solution**:

- Comprehensive synonym handling in Claude's system prompt
- Semantic embeddings that naturally cluster related concepts
- Business intelligence training for contextual understanding

### Challenge 2: Formula Interpretation

**Problem**: `=B5/B6` could mean anything without context.

**Solution**:

- Column header analysis for semantic context
- Formula + context combination in embeddings
- Claude's business intelligence for formula meaning interpretation

### Challenge 3: Cross-Sheet Complexity

**Problem**: Spreadsheets with complex cross-sheet references are hard to understand.

**Solution**:

- Regex-based cross-sheet reference extraction
- Knowledge graph representation of dependencies
- Context propagation across sheets in results

### Challenge 4: Performance with Large Spreadsheets

**Problem**: Real-time processing of large spreadsheets is slow.

**Solution**:

- Intelligent caching at multiple levels
- Precomputed embeddings with fast similarity search
- Batch processing and lazy loading
- Progressive disclosure of complexity

### Challenge 5: Natural Language Query Ambiguity

**Problem**: "Find profit" could mean many different things.

**Solution**:

- Rich context building with sample values and statistics
- Claude's sophisticated prompt engineering for disambiguation
- Multiple result types (columns AND rows) for comprehensive coverage

---

## Future Enhancements

### 1. Advanced Analytics Integration

- **Time Series Analysis**: Detect and analyze temporal patterns
- **Trend Detection**: Identify growth/decline patterns automatically
- **Anomaly Detection**: Highlight unusual values or patterns
- **Predictive Insights**: Basic forecasting based on historical data

### 2. Enhanced Visualization

- **Interactive Charts**: Dynamic visualizations of search results
- **Relationship Graphs**: Visual representation of cross-sheet dependencies
- **Formula Flow Diagrams**: Visual formula dependency trees
- **Business Dashboards**: Auto-generated KPI dashboards

### 3. Collaboration Features

- **Shared Searches**: Save and share search queries
- **Comments & Annotations**: Add business context to search results
- **Version Tracking**: Track changes to spreadsheet structure
- **Team Workspaces**: Multi-user collaboration environments

### 4. Advanced AI Capabilities

- **Multi-Model Support**: Integration with GPT-4, Gemini, etc.
- **Custom Business Rules**: User-defined business logic
- **Auto-Documentation**: Generate business documentation from spreadsheets
- **Smart Recommendations**: Suggest relevant queries based on usage patterns

### 5. Enterprise Features

- **SSO Integration**: Enterprise authentication systems
- **Audit Logging**: Comprehensive usage tracking
- **Role-Based Access**: Granular permission management
- **API Endpoints**: RESTful API for integration
- **Bulk Processing**: Handle multiple spreadsheets simultaneously

### 6. Performance & Scalability

- **Distributed Processing**: Handle enterprise-scale spreadsheets
- **Real-time Updates**: Live sync with spreadsheet changes
- **Advanced Caching**: Redis/Memcached integration
- **Load Balancing**: Support for high-concurrency usage

---

## Conclusion

This semantic search engine represents a significant advancement in how users interact with spreadsheet data. By combining knowledge graph representation, advanced NLP embeddings, and sophisticated AI reasoning, it bridges the gap between human conceptual thinking and structured data analysis.

The system's architecture is designed for both immediate usability and future extensibility, with comprehensive caching, error handling, and security measures that make it suitable for both research and production environments.

The project successfully demonstrates how modern AI technologies can be applied to traditional business tools to create more intuitive and powerful user experiences.
