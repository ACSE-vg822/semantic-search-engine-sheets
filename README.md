# Semantic Search Engine for Spreadsheets

A semantic search engine that understands spreadsheet content conceptually, allowing users to find data using natural language queries like "find profitability metrics" or "show cost calculations" instead of exact text matches.

## Developer Setup

1. **Clone and setup environment:**

   ```bash
   git clone <repository-url>
   cd semantic-search-engine-sheets
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Google Sheets API:**

   - Create a Google Cloud project and enable Google Sheets API
   - Get your Claude API key from Anthropic
   - Create `.streamlit/secrets.toml` with this format:

   ```toml
   claude_api_key = "your-claude-api-key-here"

   [google_credentials]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
   client_email = "your-service-account@project.iam.gserviceaccount.com"
   client_id = "your-client-id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40project.iam.gserviceaccount.com"
   universe_domain = "googleapis.com"
   ```

3. **Run the application:**
   ```bash
   streamlit run app_new.py
   ```

## Contributing Areas

- **Data Ingestion** (`src/data_ingestion/`): Enhance spreadsheet parsing and knowledge graph construction
- **RAG Retriever** (`src/rag/`): Improve semantic similarity and embedding strategies
- **Search Engine** (`src/semantic_search/`): Expand natural language query processing and business concept recognition
- **UI/UX**: Enhance the Streamlit interface for better result visualization and user experience
