from typing import Dict

class Explainer:
    def why(self, resource: Dict, profile, user_query: str) -> str:
        tags = resource.get("tags") or []
        top_tags = set(getattr(profile, "top_tags", []) or [])
        if top_tags and any(t in top_tags for t in tags):
            t = next(t for t in tags if t in top_tags)
            return f"Because you've been exploring **{t}**, this digs deeper via “{resource.get('title','')}”."
        q_kw = user_query.strip()[:60]
        return f"Directly related to your question “{q_kw}…”, and complements your recent activity."