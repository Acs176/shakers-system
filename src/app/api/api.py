import os

from loguru import logger
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio

from src.metrics_setup import init_metrics
from src.logging_setup import setup_logging, span

from src.app.data_ingestor.vector_index import FaissVectorStore
from src.app.data_ingestor.resource_index import load_resource_index
from src.app.user.db import init_db, create_profile, get_profile, record_query, append_seen
from src.app.recommender.orchestrator import Recommender
from src.app.rag.orchestrator import RagOrchestrator

## API OBJECTS
class AskRequest(BaseModel):
    q: str = Field(..., description="User query")
    uid: Optional[str] = Field(None, description="User ID; if omitted, a new profile is created")

class AskResponse(BaseModel):
    answer: Dict[str, Any] = Field(..., description="Grounded RAG response payload as returned by RagOrchestrator")
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommended resources")
    meta: Dict[str, Any] = Field(..., description="Aux metadata (timings, model/provider info, profile id, etc.)")

RESOURCE_JSON = "./kb/resource_catalog.json"

# ------------------------
# App setup
# ------------------------

def create_app() -> FastAPI:
    load_dotenv()
    setup_logging(app_name="rag_minimal_api")
    METRICS = init_metrics("rag_minimal_api")

    app = FastAPI(
        title="RAG Minimal API",
        version="1.0.0",
        description="Simple API layer that exposes /ask to run the existing RAG + recommender pipeline."
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # These will be set during startup and injected via dependency
    state: Dict[str, Any] = {}

    @app.on_event("startup")
    def _startup():
        # Init persistent components once
        with span("startup"):
            with span("vector_index.load"):
                index_path = os.getenv("VECTOR_INDEX_PATH") or "./rag_index"
                state["vx"] = FaissVectorStore.load(index_path)

            with span("resource_index.load"):
                state["ri"] = load_resource_index(RESOURCE_JSON)

            with span("user_db.init"):
                init_db()

            oos_threshold = float(os.getenv("OOS_THRESHOLD", "0.35"))
            llm_provider = os.getenv("LLM_PROVIDER")
            gemini_key = os.getenv("GEMINI_API_KEY")

            state["recommender"] = Recommender(state["ri"])
            state["rag_orch"] = RagOrchestrator(llm_provider, gemini_key, state["vx"], oos_threshold)

            logger.info("Startup complete. Index, resources, DB, and orchestrator ready.")

    def deps():
        if not state:
            raise HTTPException(status_code=503, detail="Service not ready")
        return state

    # ------------------------
    # /ask endpoint
    # ------------------------
    @app.post("/ask", response_model=AskResponse, summary="Run RAG + Recommendations")
    async def ask(payload: AskRequest, dep=Depends(deps)):
        try:
            with span("handler.ask"):
                # Profiles
                profile = get_profile(payload.uid) if payload.uid else None
                if profile is None:
                    profile = create_profile()  
                
                with span("query.run"):
                    rec_task = asyncio.to_thread(dep["recommender"].recommend, profile, payload.q)
                    rag_task = asyncio.to_thread(dep["rag_orch"].get_grounded_response, payload.q)
                    recs, answer = await asyncio.gather(rec_task, rag_task)

                # Persist
                record_query(profile, payload.q)
                append_seen(profile, [r["id"] for r in recs])

                #TODO: METRICS.increment("ask_requests_total")

                return AskResponse(
                    answer=answer,
                    recommendations=recs,
                    meta={
                        "profile_id": profile.user_id,
                        "provider": os.getenv("LLM_PROVIDER"),
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unhandled error in /ask")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", include_in_schema=False)
    def health():
        return {"ok": True}

    return app
