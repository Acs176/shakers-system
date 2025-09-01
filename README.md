# Intelligent Technical Support System (Case Study)

This project implements an intelligent technical support system with personalized recommendations for the **Shakers** platform.  
It includes:
- A **RAG (Retrieval-Augmented Generation) query service** for answering user questions from a knowledge base.
- A **personalized recommendation service** that suggests relevant resources based on user history.

---

## Getting Started

1. Create a Virtual Environment

It’s recommended to use a virtual environment to isolate dependencies. I DO NOT INCLUDE MY VENV due to large files and it's in general a bad practice.
Creating a virtual environment and installing the dependencies takes around 2 minutes:
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. ENVIRONMENT VARIABLES

I included a `.env.example` file. (The OTEL variables are NOT necessary in order to run the app)

```bash
GEMINI_API_KEY=""                           ## Your API key
LLM_PROVIDER="gemini"
OTEL_RESOURCE_ATTRIBUTES=""                 ## NO NEED TO FILL
OTEL_EXPORTER_OTLP_ENDPOINT=""              ## NO NEED TO FILL
OTEL_EXPORTER_OTLP_HEADERS=""               ## NO NEED TO FILL
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" ## NO NEED TO FILL
VECTOR_INDEX_PATH=""                        ## The path where you saved your vector index
KNOWLEDGE_BASE_PATH="./kb"                  ## leave this one if you don't change the default path
```

4. Build the vector store

The documents are stored in the /kb folder, you just need to indicate the path where the index will end up (I used "/rag_index" as default in the rest of the code).
```bash
python -m src.app.data_ingestor.build_index_script ./kb ./index_output_path
```

5. Run the Application
```bash
python -m src.main
```

## Making requests
Use CURL or Postman to hit the backend on localhost:8000. There's an /ask endpoint exposed:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "do you have AI developers?",
    "uid": "u_377e7b"
  }' # Linux

curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{ \"q\": \"do you have AI developers?\", \"uid\": \"u_377e7b\" }" # Windows (can't paste multiline)
```
📂 Project Structure
```bash
src/
├── app/
│   ├── api/             # API setup (FastAPI)
│   ├── data_ingestor/   # Data ingestion & preprocessing
│   ├── rag/             # Retrieval-Augmented Generation logic
│   ├── recommender/     # Personalized recommendation engine
│   └── user/            # User db & profile management
├── args_handler.py      # CLI args & configuration
├── logging_setup.py     # Logging configuration
├── metrics_setup.py     # Metrics & monitoring
└── main.py              # Entry point
```
⚙️ Technologies Used
- Python 3.9+

- FastAPI (backend)

- OTEL+Grafana (dashboard) [unfinished]

- Custom retrieval + orchestration backend

- FAISS (vector db)

