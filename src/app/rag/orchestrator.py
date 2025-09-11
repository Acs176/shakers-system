from typing import Dict, Optional, List
from src.app.rag.llm_router import LlmRouter
from src.app.rag.retriever import Retriever
from src.logging_setup import span

class RagOrchestrator:
    retriever: Retriever
    llm_router: LlmRouter

    def __init__(self, llm_provider, api_key, vx, oos_threshold):
        self.retriever = Retriever(vx, oos_threshold)
        self.llm_router = LlmRouter(llm_provider, api_key)
    
    def get_grounded_response(self, query: str, include_contexts: bool = False) -> Dict:
        with span("rag.run"):
            context, out_of_scope, citations, ctx_texts = self.retriever.get_top_results(query, k=3, return_texts=include_contexts) ## TODO: Magic number
            if out_of_scope:
                answer = "I don't have information on this in the current knowledge base."
                citations = []
                ctx_texts = [] if include_contexts else None
            else:
                answer = self.llm_router.generate_answer(query, context)
            payload = {
                "answer": answer,
                "citations": citations
            }
            if include_contexts:
                payload["contexts"] = ctx_texts or []
            return payload
