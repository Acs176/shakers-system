import argparse
import os
from typing import List

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from langchain_community.embeddings import HuggingFaceEmbeddings


def _make_llm(provider: str, api_key: str | None = None):
    provider = (provider or "gemini").lower()
    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. pip install langchain-google-genai")
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
        if not key:
            raise ValueError("Missing Google API key. Provide --api_key or set GOOGLE_API_KEY.")
        os.environ.setdefault("GOOGLE_API_KEY", key)
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. pip install langchain-openai")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OpenAI API key. Provide --api_key or set OPENAI_API_KEY.")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=key)
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Use 'gemini' or 'openai'.")


def compute_faithfulness(question: str, answer: str, contexts: List[str], provider: str = "gemini", api_key: str | None = None) -> float:
    """
    Compute RAGAS faithfulness score for a single QA item.
    Inputs:
      - question: user question
      - answer: model answer
      - contexts: list of retrieved context strings used to answer
      - provider: 'gemini' or 'openai'
      - api_key: optional API key (falls back to env vars)
    Returns a float in [0,1].
    """
    if not contexts:
        raise ValueError("'contexts' must contain at least one context string")

    llm = _make_llm(provider, api_key)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    res = ragas_evaluate(ds, metrics=[faithfulness], llm=llm, embeddings=emb)

    # res is a datasets.Dataset with a 'faithfulness' column
    try:
        score = float(res[0]["faithfulness"])  # type: ignore[index]
    except Exception:
        # Fallback: try to convert to pandas
        try:
            df = res.to_pandas()  # type: ignore[attr-defined]
            score = float(df.loc[0, "faithfulness"])  # type: ignore[index]
        except Exception as e:
            raise RuntimeError(f"Failed to extract faithfulness score: {e}")
    return max(0.0, min(1.0, score))


def main():
    ap = argparse.ArgumentParser(description="Compute RAGAS faithfulness for a single QA+contexts item")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--answer", required=True, help="Model answer")
    ap.add_argument("--context", dest="contexts", action="append", required=True, help="One context string (repeatable)")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="LLM provider for RAGAS")
    ap.add_argument("--api_key", default=None, help="API key for provider (or set env var)")
    args = ap.parse_args()

    score = compute_faithfulness(args.question, args.answer, args.contexts, provider=args.provider, api_key=args.api_key)
    print(f"Faithfulness: {score:.4f}")


if __name__ == "__main__":
    main()

