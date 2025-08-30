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
from concurrent.futures import ThreadPoolExecutor


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
    vx = VectorIndex.load(args.index) ## TODO: Handle error
    ri = load_resource_index(RESOURCE_JSON)
    ## TODO: replace with userDB
    with open("./kb/sample_user_profiles.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)
    user = Profile.from_dict(profiles[0])
    if is_empty_profile(user):
        user = Profile.create()

    recommender = Recommender(ri)
    rag_orch = RagOrchestrator(os.getenv("LLM_PROVIDER"), os.getenv("GEMINI_API_KEY"), vx, args.oos_threshold)
    recommendations = recommender.recommend(user, query)
    resp = rag_orch.get_grounded_response(query)
   
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    print(json.dumps(recommendations, ensure_ascii=False, indent=2))

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