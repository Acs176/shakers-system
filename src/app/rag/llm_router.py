import re, json
from typing import List, Dict, Optional
import google.generativeai as genai
from openai import OpenAI
from loguru import logger

## HELPER
def _parse_json_array(text: str):
    s = text.strip()
    try:
        arr = json.loads(s); return arr
    except: pass
    # strip code fences + grab the first [...] span
    s = re.sub(r"^```[\w]*\s*|\s*```$", "", s, flags=re.DOTALL)
    i, j = s.find("["), s.rfind("]")
    if i != -1 and j != -1:
        try: return json.loads(s[i:j+1])
        except: pass
    # last resort: grab quoted strings
    return re.findall(r'"([^"]+)"', s)[:3]

class LlmRouter:
    _llm_provider: str
    _api_key: str
    
    def __init__(self, llm_provider, api_key):
        self._llm_provider = llm_provider
        self._api_key = api_key

    def _build_prompt(self, query: str, docs: List[Dict]) -> str:
        parts = []
        for d in docs:
            parts.append(
                f"{d['text']}\nSOURCE: {d['title']} — {d['section']} — {d['source']} — {d['id']}\n"
            )
        context = "\n\n".join(parts)
        system = (
            "You are a helpful support assistant for the Shakers platform.\n"
            "Answer ONLY using the provided context.\n"
            "Be concise and include a 'Sources:' list referencing the SOURCE lines.\n"
        )
        return f"{system}\nContext:\n{context}\n\nUser question:\n{query}\n\nAnswer:"
    
    def _try_gemini(self, prompt: str) -> Optional[str]:
        try:
            if not self._api_key:
                return None
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            logger.error("llm.error", e)
            return None

    def _try_openai(self, prompt: str) -> Optional[str]:
        try:
            if not getattr(self, "_api_key", None):
                return None
            client = OpenAI(api_key=self._api_key)
            resp = client.responses.create(
                model="gpt-4o-mini",   # low-cost general text model
                input=prompt,
                temperature=0,         # deterministic for caching
                max_output_tokens=512, # tweak as you like
            )
            text = getattr(resp, "output_text", None)
            return text.strip() if text else None
        except Exception:
            logger.exception("llm.error")
            return None

    def generate_answer(self, query: str, docs: List[Dict]) -> str:
        prompt = self._build_prompt(query, docs)
        text = None
        if self._llm_provider == "gemini":
            text = self._try_gemini(prompt)
        if self._llm_provider == "openai":
            text = self._try_openai(prompt)
        if text is None: ## if LLMs fail
            text = self.extractive_fallback(docs)
        return text

    

    def extractive_fallback(self, docs: List[Dict]) -> str:
        """
        Simple extractive answer: return a concise summary (first 3-5 sentences)
        from the top doc(s) + a Sources list. Works without an LLM.
        """
        joined = " ".join(d["text"] for d in docs)
        sentences = re.split(r'(?<=[\.\!\?])\s+', joined)
        summary = " ".join(sentences[:5]).strip()
        lines = []
        lines.append(summary if summary else "I don't have that information in the provided context.")
        lines.append("\nSources:")
        for d in docs:
            lines.append(f"- {d['title']} — {d['section']} ({d['source']}#{d['id'].split('#')[-1]})")
        return "\n".join(lines).strip()

    def rewrite_queries(self, user_query: str, n: int = 3) -> List[str]:
        """
        Ask the LLM to produce n diverse, intent-preserving rewrites for retrieval.
        """
        prompt = f"""DO NOT RESPOND WITH MARKDOWN. ONLY VALID JSON. You are a query optimizer for a retrieval-augmented generation (RAG) system over internal product documentation.

        Rewrite the user's question into {n} diverse alternative search queries that:
        - preserve the original intent and scope,
        - expand likely domain terms and synonyms,
        - avoid adding facts not in the question,
        - keep each query under 12 words,
        - do not number the items.

        Return ONLY a JSON array of strings. NO MARKDOWN. ONLY JSON

        User question: '{user_query}'
        """

        text = self._try_gemini(prompt)
        arr = _parse_json_array(text) if text else None
        if arr and len(arr) >= 1:
            return arr[:n]