from typing import Dict
from src.app.rag.llm_router import LlmRouter
from src.app.rag.retriever import Retriever
from src.logging_setup import span

class RagOrchestrator:
    retriever: Retriever
    llm_router: LlmRouter

    def __init__(self, llm_provider, api_key, vx, oos_threshold):
        self.retriever = Retriever(vx, oos_threshold)
        self.llm_router = LlmRouter(llm_provider, api_key)
    
    def get_grounded_response(self, query: str) -> Dict:
        with span("rag.run"):
            context, out_of_scope, citations = self.retriever.get_top_results(query, k=3) ## TODO: Magic number
            if out_of_scope:
                answer = "I don't have information on this in the current knowledge base."
                citations = []
            else:
                answer = self.llm_router.generate_answer(query, context)
            return {
                "answer": answer,
                "citations": citations
            }
