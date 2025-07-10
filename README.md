# Semantic Search Engine for Spreadsheets

A semantic search engine that lets you query spreadsheet data using natural language instead of exact text matches.

## Setup

```bash
git clone <repository-url>
cd semantic-search-engine-sheets
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate
pip install -r requirements.txt
python setup.py install
```

Create `.streamlit/secrets.toml` with your Claude API key and Google Sheets credentials:

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

Then run:

```bash
streamlit run streamlit_app.py
```
