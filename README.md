# Semantic Search Engine for Spreadsheets

An intelligent spreadsheet search system that understands business concepts and allows users to find information using natural language queries rather than exact text matches.

## 🎯 Project Overview

This system bridges the gap between how users think about spreadsheet data (conceptually) and how traditional search works (structurally). Instead of searching for exact text matches, users can ask semantic questions like:

- _"Find all profitability metrics"_
- _"Show me cost calculations"_
- _"Where are my growth rates?"_
- _"Find efficiency ratios"_

The system understands business concepts, interprets context, and provides intelligent responses with statistical analysis.

## 🏗️ Architecture & Code Flow

### 1. Core Components

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

### 2. Detailed Flow Explanation

#### **Phase 1: Data Ingestion & Knowledge Graph Construction**

**File: `src/data_ingestion/spreadsheet_parser_advance.py`**

1. **Google Sheets Connection**:

   - Authenticates using service account credentials from Streamlit secrets
   - Connects to specified spreadsheet via Google Sheets API

2. **Data Parsing**:

   - Extracts all cell values and formulas from each worksheet
   - Identifies data types (text, number, formula)
   - Captures cell addresses and cross-sheet references

3. **Knowledge Graph Building**:
   ```python
   SpreadsheetData → SpreadsheetKnowledgeGraph
   ```
   - **Column Metadata**: Headers, data types, sample values, formulas
   - **Cross-Sheet References**: Tracks dependencies between sheets
   - **Statistical Context**: Addresses ranges for later analysis

#### **Phase 2: Semantic Embedding & Retrieval Setup**

**File: `src/rag/retriever.py`**

1. **Corpus Building**:

   - Creates searchable text from column headers, data types, and sample values
   - Each entry: `"Header (data_type) sample_value1 sample_value2..."`

2. **Embedding Generation**:

   - Uses Sentence Transformers (`all-MiniLM-L6-v2`) for semantic embeddings
   - Converts text corpus into vector representations
   - Falls back to text similarity if embeddings unavailable

3. **Retrieval Engine**:
   - Semantic similarity search using cosine similarity
   - Returns top-k most relevant columns with confidence scores

#### **Phase 3: Query Processing & Enhancement**

**File: `src/semantic_search/search_engine_advanced.py`**

1. **Query Analysis**:

   - Retrieves top 5 most relevant columns using the RAG retriever
   - Fetches real-time data from Google Sheets for statistical analysis

2. **Statistical Enhancement**:

   ```python
   def _calculate_numerical_stats(values):
       return {
           'min': min_value,
           'max': max_value,
           'mean': average,
           'count': data_points,
           'total_cells': all_cells
       }
   ```

3. **Context Building**:

   - Combines column metadata with fresh statistical data
   - Creates comprehensive JSON context for AI analysis

4. **Claude AI Integration**:
   - Sends enhanced context to Claude 3.7 Sonnet
   - Uses structured system prompts for business intelligence
   - Returns formatted JSON responses with explanations

#### **Phase 4: User Interface & Caching**

**File: `app_new.py`**

1. **Streamlit Interface**:

   - Spreadsheet selection (Sales Dashboard, Financial Model)
   - Query input and result display
   - Progress indicators and error handling

2. **Intelligent Caching**:

   ```python
   # Session state caching
   st.session_state.knowledge_graph
   st.session_state.sentence_transformer_model
   st.session_state.retriever
   ```

   - Knowledge graphs cached per spreadsheet
   - Sentence transformer models cached globally
   - Pre-computed embeddings stored for performance

3. **Result Processing**:
   - Parses Claude's JSON responses
   - Displays results with business context
   - Shows formulas, explanations, and cross-sheet references

## 🔧 Technical Implementation

### Data Structures

```python
@dataclass
class ColumnMetadata:
    header: str                    # Column header name
    data_type: str                # 'text', 'number', 'formula'
    first_cell_formula: str       # Formula if applicable
    sheet: str                    # Sheet name
    sample_values: List           # First 5 data values
    addresses: str               # Cell range (e.g., "A2:A13")
    cross_sheet_refs: List[str]  # Referenced sheet names

@dataclass
class SpreadsheetKnowledgeGraph:
    title: str                    # Spreadsheet title
    sheets: Dict[str, SheetMetadata]  # Sheet name → metadata
```

### Key Algorithms

1. **Cross-Sheet Reference Detection**:

   ```python
   CROSS_SHEET_REGEX = re.compile(r"(?:'([^']+)')|(?:(\w+)![$]?[A-Z]+[$]?\d+)")
   ```

2. **Semantic Similarity Scoring**:

   ```python
   scores = util.pytorch_cos_sim(query_embedding, column_embeddings)
   ```

3. **Statistical Analysis**:
   - Real-time data fetching from Google Sheets
   - Numerical value extraction and analysis
   - Min/max/mean calculations for business insights

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

**Required Dependencies**:

- `streamlit` - Web interface
- `gspread` - Google Sheets API
- `anthropic` - Claude AI integration
- `sentence-transformers` - Semantic embeddings
- `google-auth` - Authentication

### Configuration

1. **Google Sheets API**:

   - Set up service account credentials
   - Add to `.streamlit/secrets.toml`:

   ```toml
   [google_credentials]
   # Your service account JSON here
   ```

2. **Claude API**:
   ```toml
   claude_api_key = "your_claude_api_key"
   ```

### Running the Application

```bash
streamlit run app_new.py
```

## 🎯 Usage Examples

### Supported Query Types

1. **Conceptual Queries**:

   - `"Find all profitability metrics"` → Gross margin, net profit, EBITDA
   - `"Show cost calculations"` → COGS, expenses, overhead
   - `"Where are my growth rates?"` → YoY%, QoQ%, CAGR formulas

2. **Functional Queries**:

   - `"Show percentage calculations"` → All percentage formulas
   - `"Find average formulas"` → AVERAGE, SUM/COUNT combinations
   - `"What conditional calculations exist?"` → IF, SUMIF, COUNTIF

3. **Statistical Queries**:
   - `"Revenue ranging from X to Y"` → Returns numerical ranges
   - `"Sales data with statistics"` → Min/max/average analysis

### Result Format

```json
[
  {
    "concept_name": "Gross Profit Margin",
    "sheet": "P&L Statement",
    "header": "Gross Margin %",
    "cell_range": "D15:D18",
    "formula": "=(Revenue-COGS)/Revenue",
    "explanation": "Direct margin calculation using standard formula"
  }
]
```

## 🔍 Business Intelligence Features

### Advanced Analytics

- **Statistical Enhancement**: Real-time min/max/mean calculations
- **Cross-Sheet Tracking**: Identifies relationships between worksheets
- **Formula Interpretation**: Understands business meaning of calculations
- **Context-Aware Ranking**: Prioritizes relevant business metrics

### Performance Optimizations

- **Knowledge Graph Caching**: Prevents redundant parsing
- **Embedding Pre-computation**: Cached semantic vectors
- **Model Persistence**: Reuses loaded transformer models
- **Incremental Updates**: Efficient data refresh strategies

## 🏢 Business Context Understanding

The system recognizes common business patterns:

- **Financial Metrics**: Revenue, profit, margins, ratios
- **Operational KPIs**: Efficiency, productivity, growth rates
- **Analytical Functions**: Averages, sums, comparisons
- **Time Series**: Monthly, quarterly, yearly progressions
- **Benchmarking**: Budget vs actual, variance analysis

## 🛠️ Extending the System

### Adding New Data Sources

1. Implement parser interface in `data_ingestion/`
2. Update knowledge graph structure if needed
3. Modify retriever to handle new metadata

### Enhancing Search Capabilities

1. Add domain-specific embeddings in `rag/retriever.py`
2. Implement custom similarity metrics
3. Extend statistical analysis in `search_engine_advanced.py`

### UI Improvements

1. Add visualization components in `app_new.py`
2. Implement result filtering and sorting
3. Create export functionality for search results

This semantic search engine transforms how users interact with spreadsheet data, making business intelligence more accessible through natural language understanding.
