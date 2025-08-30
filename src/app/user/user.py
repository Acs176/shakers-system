# -----------------------------
# --- PROFILE -----------------
# -----------------------------
import uuid, time
from typing import List, Dict, Optional
from dataclasses import field, dataclass
from datetime import datetime
from copy import deepcopy

@dataclass
class Query:
    ts: datetime
    text: str
    tags: List[str]


@dataclass
class Profile:
    user_id: str
    name: str
    locale: str
    created_at: datetime
    last_active: datetime
    query_history: List[Query]
    seen_resource_ids: List[str]
    clicked_resource_ids: List[str]
    top_tags: List[str]

    @classmethod
    def create(cls, name: str = None, locale: str = "en") -> "Profile":
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return cls(
            user_id= f"u_{uuid.uuid4().hex[:6]}",
            name= name or "Anonymous",
            locale= locale,
            created_at= now,
            last_active= now,
            query_history= [],
            seen_resource_ids= [],
            clicked_resource_ids= [],
            top_tags= [],
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            user_id=d.get("user_id") or f"u_{uuid.uuid4().hex[:6]}",
            name=d.get("name") or "Anonymous",
            locale=d.get("locale", "en"),
            created_at=d.get("created_at", ""),
            last_active=d.get("last_active", ""),
            query_history=[Query(**q) for q in d.get("query_history", [])],
            seen_resource_ids=list(d.get("seen_resource_ids", [])),
            clicked_resource_ids=list(d.get("clicked_resource_ids", [])),
            top_tags=list(d.get("top_tags", [])),
        )
        
def is_empty_profile(p: Optional[Profile]) -> bool:
        # Treat None, {} or profiles without any history/seen items as "empty"
        return (not p) or (not p.query_history and not p.seen_resource_ids)

def profile_update_after_recs(user_profile: Profile, recommendations: List[Dict], user_query: str) -> Dict:
    """
    Returns a minimal mutation you can persist after responding.
    """
    seen = set(user_profile.seen_resource_ids or [])
    for r in recommendations:
        seen.add(r["id"])
    return {
        "append_query_history": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "text": user_query, "tags": []},
        "seen_resource_ids": sorted(seen),
    }

def update_profile(
    profile: Profile,
    delta: Dict,
    query: str = None,
    query_tags: Optional[List[str]] = None
    ) -> Profile:
    """
    Apply the 'profile_update' delta, and ensure the current
    question is recorded even if no delta was provided.
    Returns a NEW profile dict.
    """
    p = deepcopy(profile) if profile else create_profile()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if delta.get("append_query_history"):
        p.query_history.append(delta["append_query_history"])
    if delta.get("seen_resource_ids"):
        seen = set(p.seen_resource_ids)
        seen.update(delta["seen_resource_ids"])
        p.seen_resource_ids = sorted(seen)

    
    if not p.query_history or p.query_history[-1].get("text") != query:
        p.query_history.append({
            "ts": now,
            "text": query,
            "tags": query_tags or []
        })


    # if query_tags:
        ## TODO: UPDATE TOP TAGS

    p["last_active"] = now
    return p