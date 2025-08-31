import sys, os, json

from dotenv import load_dotenv
from src.logging_setup import setup_logging, span
from src.metrics_setup import init_metrics, time_histogram
from src.app.data_ingestor.vector_index import VectorIndex, build_index
from src.app.rag.orchestrator import RagOrchestrator
from src.app.data_ingestor.resource_index import load_resource_index
from src.app.recommender.orchestrator import Recommender
from src.args_handler import setup_args
from src.app.user.user import (
    Profile,
    is_empty_profile,
)
from src.app.user.db import init_db, create_profile, get_profile, record_query, append_seen


RESOURCE_JSON = "./kb/resource_catalog.json"

if __name__ == "__main__":
    load_dotenv()
    setup_logging(app_name="rag_minimal")
    METRICS = init_metrics("rag_minimal")

    args = setup_args()
    if args.cmd == "index":
        with span("index.build"):
            build_index(args.kb, args.out, chunk_chars=args.chunk_chars, overlap=args.overlap, model=args.model)
            sys.exit() ## TODO: Temporal

    query = args.q
    ## Load DBs
    with span("vector_index.load"):
        vx = VectorIndex.load(args.index) ## TODO: Handle error
    
    ri = load_resource_index(RESOURCE_JSON)
    with span("user_db.init"):
        init_db() ## userDB

    profile = get_profile(args.uid)
    if profile is None:
        profile = create_profile()

    recommender = Recommender(ri)
    rag_orch = RagOrchestrator(os.getenv("LLM_PROVIDER"), os.getenv("GEMINI_API_KEY"), vx, args.oos_threshold)
    with span("recommend.run"):
        recommendations = recommender.recommend(profile, query)
    with span("rag.run"):
        resp = rag_orch.get_grounded_response(query)
   
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    print(json.dumps(recommendations, ensure_ascii=False, indent=2))

    record_query(profile, query)
    append_seen(profile, [rec["id"] for rec in recommendations])

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