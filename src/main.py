from dotenv import load_dotenv
from src.logging_setup import setup_logging, span
from src.metrics_setup import init_metrics, time_histogram
from src.app.data_ingestor.vector_index import VectorIndex, build_index
from src.app.rag.orchestrator import RagOrchestrator
from src.args_handler import setup_args
import sys, os, json


if __name__ == "__main__":
    load_dotenv()
    setup_logging(app_name="rag_minimal")
    METRICS = init_metrics("rag_minimal")

    args = setup_args()

    RESOURCE_JSON = "./kb/resource_catalog.json"
    if args.cmd == "index":
        with span("index.build"):
            build_index(args.kb, args.out, chunk_chars=args.chunk_chars, overlap=args.overlap, model=args.model)
            sys.exit() ## TODO: Temporal

    ## Load VectorIndex
    vx = VectorIndex.load(args.index) ## TODO: Handle error
    rag_orch = RagOrchestrator(os.getenv("LLM_PROVIDER"), os.getenv("GEMINI_API_KEY"), vx, args.oos_threshold)
    resp = rag_orch.get_grounded_response(args.q)
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    raise NotImplementedError
    ## load envs
    ## init logs
    ## init metrics
    ## cmd args ...

    ## server.run()
        ## init DBs
            ## init vector store
            ## init UserDB
            ## init resourceIndex
        ## rag.ask(query, METRICS)
        ## recommender.recommend(query, user_id, METRICS)