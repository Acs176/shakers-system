import json, sys
from typing import Iterable, List, Dict
import datetime
from src.app.user.user import Profile, Query
from src.app.user.db import (
    init_db, _conn, _iso_z_to_dt, _dt_to_iso_z
)

def parse_iso_z(s: str) -> datetime:
    return _iso_z_to_dt(s)

def import_profile(profile: Profile) -> None:
    """
    Insert/update the base profile row using its existing created_at/last_active.
    Does NOT touch related lists; call the other import_* helpers next.
    """
    with _conn() as con:
        con.execute(
            """
            INSERT INTO profiles (user_id, name, locale, created_at, last_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              name = excluded.name,
              locale = excluded.locale,
              last_active = excluded.last_active
            """,
            (
                profile.user_id,
                profile.name,
                profile.locale,
                _dt_to_iso_z(profile.created_at),
                _dt_to_iso_z(profile.last_active),
            ),
        )

def import_query(profile: Profile, q: Query) -> None:
    """Insert a query with an explicit timestamp and tags. Idempotent via UNIQUE index."""
    with _conn() as con:
        cur = con.cursor()
        # Try insert; ignore if duplicate (user_id, ts, text) already exists.
        cur.execute(
            """
            INSERT OR IGNORE INTO queries (user_id, ts, text)
            VALUES (?, ?, ?)
            """,
            (profile.user_id, _dt_to_iso_z(q.ts), q.text),
        )
        # If it already existed, fetch its id; otherwise use lastrowid.
        if cur.lastrowid:
            qid = cur.lastrowid
        else:
            row = cur.execute(
                "SELECT id FROM queries WHERE user_id=? AND ts=? AND text=?",
                (profile.user_id, _dt_to_iso_z(q.ts), q.text),
            ).fetchone()
            qid = row["id"]
        if q.tags:
            cur.executemany(
                "INSERT OR IGNORE INTO query_tags (query_id, tag) VALUES (?, ?)",
                [(qid, t) for t in q.tags],
            )

def import_seen(profile: Profile, resource_ids: Iterable[str]) -> None:
    ids = [(profile.user_id, rid) for rid in set(resource_ids)]
    if not ids: return
    with _conn() as con:
        con.executemany(
            "INSERT OR IGNORE INTO seen_resources (user_id, resource_id) VALUES (?, ?)",
            ids,
        )

def import_clicked(profile: Profile, resource_ids: Iterable[str]) -> None:
    ids = [(profile.user_id, rid) for rid in set(resource_ids)]
    if not ids: return
    with _conn() as con:
        con.executemany(
            "INSERT OR IGNORE INTO clicked_resources (user_id, resource_id) VALUES (?, ?)",
            ids,
        )

def import_top_tags(profile: Profile, tags_in_order: Iterable[str]) -> None:
    # Optional: if you want import to be idempotent and keep the existing snapshot when re-running,
    # use INSERT OR IGNORE. If you prefer "latest import wins", DELETE then INSERT.
    with _conn() as con:
        # "latest import wins":
        con.execute("DELETE FROM top_tags WHERE user_id = ?", (profile.user_id,))
        con.executemany(
            "INSERT INTO top_tags (user_id, tag, rank) VALUES (?, ?, ?)",
            [(profile.user_id, t, i+1) for i, t in enumerate(tags_in_order)],
        )

def _p(d: Dict) -> Profile:
    # Convert one JSON profile dict → Profile object with real datetimes & Query objects
    qhist = [
        Query(
            ts=parse_iso_z(q["ts"]),
            text=q["text"],
            tags=list(q.get("tags", [])),
        )
        for q in d.get("query_history", [])
    ]
    return Profile(
        user_id=d["user_id"],
        name=d.get("name") or "Anonymous",
        locale=d.get("locale", "en"),
        created_at=parse_iso_z(d["created_at"]),
        last_active=parse_iso_z(d["last_active"]),
        query_history=qhist,
        seen_resource_ids=list(d.get("seen_resource_ids", [])),
        clicked_resource_ids=list(d.get("clicked_resource_ids", [])),
        top_tags=list(d.get("top_tags", [])),
    )

def seed_users_once(users: List[Dict]) -> None:
    """
    Idempotent: you can run this multiple times.
    - Profiles upserted by user_id
    - Queries deduped by UNIQUE(user_id, ts, text)
    - Seen/Clicked deduped by PK(user_id, resource_id)
    - top_tags replaced atomically (latest import wins)
    """
    init_db()

    for d in users:
        profile = _p(d)

        # Upsert the base profile row
        import_profile(profile)

        # Import history (deduped)
        for q in profile.query_history:
            import_query(profile, q)

        # Import lists (deduped)
        import_seen(profile, profile.seen_resource_ids)
        import_clicked(profile, profile.clicked_resource_ids)

        # Import top tags snapshot (replace each time)
        import_top_tags(profile, profile.top_tags)

if __name__ == "__main__":
    # Usage:
    #   python src/app/user/seeder.py ./kb/sample_user_profiles.json
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        users = json.load(f)
    seed_users_once(users)
    print("✅ Seeding complete.")