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
3. Run the Application
```bash
python -m src.main
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

