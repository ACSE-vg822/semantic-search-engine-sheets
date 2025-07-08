# **Superjoin Hiring Assignment — Build Semantic Search for Spreadsheets**

<aside>
💡

Feel free to use any AI solution that you want! 🙂 But make sure you prompt it well - so that your assignment is different from others.

</aside>

## Context

At Superjoin, we believe spreadsheets are more than grids of data - they're documents full of business logic, calculations, and meaning. Users don't think in terms of cell references; they think semantically: _"Where are my profit calculations?"_, _"Show me all customer metrics"_, _"Find efficiency ratios"_.

Your task is to build a **semantic search engine** that understands spreadsheet content conceptually and allows users to find what they're looking for using natural language.

## The Problem

### How Users Actually Think vs. How Spreadsheets Work

**Users ask semantic questions:**

- _"Find all revenue calculations"_
- _"Show me cost-related formulas"_
- _"Where are my margin analyses?"_
- _"What percentage calculations do I have?"_

**Current tools only support structural queries:**

- Find cells containing "revenue"
- Show SUM formulas
- Filter by cell color
- Search exact text matches

### Your Challenge

Build a search engine that bridges this gap - one that understands what spreadsheet content means, not just what it contains.

## What You're Building

A **semantic search system** that can:

1. **Understand Business Concepts**: Recognize that "Q1 Revenue", "First Quarter Sales", and "Jan-Mar Income" all refer to similar concepts
2. **Interpret Context**: Distinguish between "Marketing Spend" (cost) vs "Marketing ROI" (efficiency metric)
3. **Find Conceptual Matches**: When someone searches "profitability", find gross margin, net profit, EBITDA calculations
4. **Handle Natural Language**: Process queries like _"show efficiency metrics"_ or _"find budget vs actual comparisons"_

## Core Features to Build

### 1️⃣ Semantic Content Understanding

**Challenge**: Make your system understand what each part of the spreadsheet represents

**Requirements:**

- **Concept Recognition**: Identify business concepts like revenue, costs, margins, ratios, forecasts
- **Synonym Handling**: Understand "sales" = "revenue", "profit" = "earnings", "efficiency" = "productivity"
- **Context Interpretation**: Recognize that =B5/B6 in a "Margin %" column calculates a margin
- **Formula Semantics**: Understand that SUM formulas in "Total Sales" calculate revenue totals

**Example Semantic Understanding:**

```
Cell Header: "Gross Profit Margin"
Formula: =(Revenue-COGS)/Revenue
→ Your system should understand: This is a profitability metric, margin calculation, percentage formula
```

### 2️⃣ Natural Language Query Processing

**Challenge**: Handle business-oriented search queries, not just keyword matching

**Query Types to Support:**

**Conceptual Queries:**

```
"Find all profitability metrics" → Gross margin, net profit, EBITDA, etc.
"Show cost calculations" → COGS, expenses, overhead allocations
"Where are my growth rates?" → YoY%, QoQ%, CAGR formulas
"Find efficiency ratios" → ROI, ROE, asset turnover, etc.
```

**Functional Queries:**

```
"Show percentage calculations" → All formulas that calculate percentages
"Find average formulas" → AVERAGE, SUM/COUNT combinations
"What conditional calculations exist?" → IF, SUMIF, COUNTIF formulas
"Show lookup formulas" → VLOOKUP, INDEX/MATCH, XLOOKUP
```

**Comparative Queries:**

```
"Budget vs actual analysis" → Variance calculations, comparison formulas
"Time series data" → Monthly, quarterly, yearly progressions
"Benchmark comparisons" → Ratios against industry standards
```

### 3️⃣ Intelligent Result Ranking & Output

**Challenge**: Return the most relevant results with meaningful context, not just cell references

**Ranking Factors:**

- **Semantic Relevance**: How closely does content match the concept?
- **Context Importance**: Is this a key metric or supporting calculation?
- **Formula Complexity**: More sophisticated calculations might be more relevant
- **Data Recency**: Recent data might be more important than historical

**Output Format Requirements:**
Your search results should provide **semantic context**, not just cell locations. Design an output format that helps users understand **what they found** and **why it's relevant**.

**Consider Including:**

- **Concept Name**: What business concept this represents
- **Location**: Where it's found (with human-readable context)
- **Value/Formula**: Current value and/or underlying calculation
- **Explanation**: Why this matches the user's query
- **Business Context**: What role this plays in the spreadsheet

**Example Output Structure for "find margin calculations":**

```
Gross Profit Margin
├── Location: 'Revenue Analysis'!C15
├── Formula: =(Revenue-COGS)/Revenue
└── Relevance: Direct margin calculation using standard formula

Operating Margin
├── Location: 'P&L Statement'!D15:D18
└── Relevance: Operating efficiency margins over time

```

**Result Presentation Options:**

- Structured JSON/objects for programmatic use
- Human-readable summaries for direct user consumption
- Grouped results by business concept rather than location

### 4️⃣ Multi-Sheet Understanding (🍪 Brownie points for this)

**Challenge**: Understand concepts across multiple spreadsheet tabs

**Requirements:**

- **Cross-Sheet Concept Tracking**: Recognize related concepts across different sheets
- **Context Switching**: Understand that "Budget" sheet and "Actuals" sheet contain related but different data
- **Relationship Recognition**: Connect forecasts, budgets, and actuals as related concept families

## Deliverables (3-4 days)\

### 1. Working Search Engine

**Core Functionality:**

- Load and parse spreadsheet content (you can use provided test sheets)
- Process natural language queries
- Return ranked, relevant results
- Handle real-time updates as content changes (🍪 Brownie points for this)

### Demo Interface Requirements

- Simple query interface (CLI, web UI, or notebook)
- Show search results with meaningful context and explanations
- Demonstrate different query types with clear output formatting
- Display semantic understanding through well-structured results

### 2. Technical Documentation

**Design Document (2-3 pages):**

- Your approach to semantic understanding
- How you handle business domain knowledge
- Query processing and result ranking methodology
- Technical architecture and data structures
- Performance considerations and trade-offs
- Challenges faced and solutions implemented

### 3. Evaluation & Testing

**Demonstrate Your System With:**

- Queries across different business domains (finance, sales, operations)
- Complex multi-concept searches
- Comparison with keyword-only search to show improvement
- Edge cases and error handling

## Test Data

Use these spreadsheets to develop and test your system:

- **Financial Model**: https://docs.google.com/spreadsheets/d/1FAQv7_hkbFKXQC-a57YfNHe89awCzyv9tLxhjR4ez0o/edit?usp=sharing
- **Sales Dashboard**: https://docs.google.com/spreadsheets/d/14eiRz4_IevXEIWcxkJHALf6jdF6DjF-TGD6FScV7aYo/edit?usp=sharing
