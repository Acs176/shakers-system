# repo.py
import sqlite3
from contextlib import contextmanager
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import uuid

from src.app.user.user import Profile, Query

DB_PATH = "./profiles.db"

# ---------- time helpers (UTC) ----------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _dt_to_iso_z(dt: datetime) -> str:
    # Ensure UTC and format like 2025-08-30T12:34:56Z
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

def _iso_z_to_dt(s: str) -> datetime:
    # Parse ...Z into timezone-aware UTC
    # Accept missing 'Z' just in case.
    if s.endswith("Z"):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    # Fallback: try plain ISO second precision
    return datetime.fromisoformat(s).astimezone(timezone.utc)

# ---------- DB plumbing ----------
@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    with _conn() as con:
        c = con.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS profiles (
          user_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          locale TEXT NOT NULL,
          created_at TEXT NOT NULL,
          last_active TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS queries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          ts TEXT NOT NULL,
          text TEXT NOT NULL,
          FOREIGN KEY (user_id) REFERENCES profiles(user_id)
        );
        CREATE TABLE IF NOT EXISTS query_tags (
          query_id INTEGER NOT NULL,
          tag TEXT NOT NULL,
          UNIQUE(query_id, tag),
          FOREIGN KEY (query_id) REFERENCES queries(id)
        );
        CREATE TABLE IF NOT EXISTS seen_resources (
          user_id TEXT NOT NULL,
          resource_id TEXT NOT NULL,
          PRIMARY KEY (user_id, resource_id),
          FOREIGN KEY (user_id) REFERENCES profiles(user_id)
        );
        CREATE TABLE IF NOT EXISTS clicked_resources (
          user_id TEXT NOT NULL,
          resource_id TEXT NOT NULL,
          PRIMARY KEY (user_id, resource_id),
          FOREIGN KEY (user_id) REFERENCES profiles(user_id)
        );
        CREATE TABLE IF NOT EXISTS top_tags (
          user_id TEXT NOT NULL,
          tag TEXT NOT NULL,
          rank INTEGER NOT NULL,
          PRIMARY KEY (user_id, tag),
          FOREIGN KEY (user_id) REFERENCES profiles(user_id)
        );
        """)

def create_profile(name: Optional[str] = None, locale: str = "en") -> Profile:
    now = _now_utc()
    return Profile(
        user_id=f"u_{uuid.uuid4().hex[:6]}",
        name=name or "Anonymous",
        locale=locale,
        created_at=now,
        last_active=now,
        query_history=[],
        seen_resource_ids=[],
        clicked_resource_ids=[],
        top_tags=[],
    )

def save_profile(profile: Profile) -> None:
    """Insert or update base profile fields (lists are stored via other helpers)."""
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

def get_profile(user_id: str) -> Optional[Profile]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return _hydrate_profile_from_row(row)

def _hydrate_profile_from_row(row: sqlite3.Row) -> Profile:
    # base fields
    p = Profile(
        user_id=row["user_id"],
        name=row["name"],
        locale=row["locale"],
        created_at=_iso_z_to_dt(row["created_at"]),
        last_active=_iso_z_to_dt(row["last_active"]),
        query_history=[],
        seen_resource_ids=[],
        clicked_resource_ids=[],
        top_tags=[],
    )
    # attach lists
    p.query_history = _load_query_history(p.user_id)
    p.seen_resource_ids = _load_ids("seen_resources", p.user_id)
    p.clicked_resource_ids = _load_ids("clicked_resources", p.user_id)
    p.top_tags = _load_top_tags(p.user_id)
    return p

def _load_ids(table: str, user_id: str) -> List[str]:
    with _conn() as con:
        rows = con.execute(
            f"SELECT resource_id FROM {table} WHERE user_id = ? ORDER BY resource_id",
            (user_id,),
        ).fetchall()
        return [r["resource_id"] for r in rows]

def _load_top_tags(user_id: str) -> List[str]:
    with _conn() as con:
        rows = con.execute(
            "SELECT tag FROM top_tags WHERE user_id = ? ORDER BY rank ASC, tag ASC",
            (user_id,),
        ).fetchall()
        return [r["tag"] for r in rows]

def _load_query_history(user_id: str, limit: int = 1000) -> List[Query]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT q.id, q.ts, q.text
            FROM queries q
            WHERE q.user_id = ?
            ORDER BY q.ts ASC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        out: List[Query] = []
        for r in rows:
            tags = con.execute(
                "SELECT tag FROM query_tags WHERE query_id = ? ORDER BY tag",
                (r["id"],),
            ).fetchall()
            out.append(Query(ts=_iso_z_to_dt(r["ts"]), text=r["text"], tags=[t["tag"] for t in tags]))
        return out

def record_query(profile: Profile, query: str, tags: Optional[List[str]] = None) -> None:
    """Persist a query and update the given Profile object in-place."""
    ts = _now_utc()
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO queries (user_id, ts, text) VALUES (?, ?, ?)",
            (profile.user_id, _dt_to_iso_z(ts), query),
        )
        qid = cur.lastrowid
        if tags:
            cur.executemany(
                "INSERT OR IGNORE INTO query_tags (query_id, tag) VALUES (?, ?)",
                [(qid, t) for t in tags],
            )
        con.execute(
            "UPDATE profiles SET last_active = ? WHERE user_id = ?",
            (_dt_to_iso_z(ts), profile.user_id),
        )
    # update in-memory
    profile.last_active = ts
    profile.query_history.append(Query(ts=ts, text=query, tags=tags or []))

def append_seen(profile: Profile, resource_ids: List[str]) -> None:
    if not resource_ids:
        return
    with _conn() as con:
        con.executemany(
            "INSERT OR IGNORE INTO seen_resources (user_id, resource_id) VALUES (?, ?)",
            [(profile.user_id, rid) for rid in resource_ids],
        )
        con.execute(
            "UPDATE profiles SET last_active = ? WHERE user_id = ?",
            (_dt_to_iso_z(_now_utc()), profile.user_id),
        )
    # update in-memory
    seen = set(profile.seen_resource_ids)
    for rid in resource_ids:
        seen.add(rid)
    profile.seen_resource_ids = sorted(seen)

def append_clicked(profile: Profile, resource_ids: List[str]) -> None:
    if not resource_ids:
        return
    with _conn() as con:
        con.executemany(
            "INSERT OR IGNORE INTO clicked_resources (user_id, resource_id) VALUES (?, ?)",
            [(profile.user_id, rid) for rid in resource_ids],
        )
        con.execute(
            "UPDATE profiles SET last_active = ? WHERE user_id = ?",
            (_dt_to_iso_z(_now_utc()), profile.user_id),
        )
    clicked = set(profile.clicked_resource_ids)
    for rid in resource_ids:
        clicked.add(rid)
    profile.clicked_resource_ids = sorted(clicked)

def set_top_tags(profile: Profile, tags_ranked: List[Tuple[str, int]]) -> None:
    """tags_ranked = [(tag, rank), ...] with rank=1 best"""
    with _conn() as con:
        con.execute("DELETE FROM top_tags WHERE user_id = ?", (profile.user_id,))
        con.executemany(
            "INSERT INTO top_tags (user_id, tag, rank) VALUES (?, ?, ?)",
            [(profile.user_id, t, r) for t, r in tags_ranked],
        )
    # update in-memory to ordered list
    profile.top_tags = [t for t, _ in sorted(tags_ranked, key=lambda x: (x[1], x[0]))]

def refresh_profile(profile: Profile) -> Profile:
    """Reload lists and timestamps from DB for this user and return the same object updated."""
    fresh = get_profile(profile.user_id)
    if fresh is None:
        return profile
    profile.name = fresh.name
    profile.locale = fresh.locale
    profile.created_at = fresh.created_at
    profile.last_active = fresh.last_active
    profile.query_history = fresh.query_history
    profile.seen_resource_ids = fresh.seen_resource_ids
    profile.clicked_resource_ids = fresh.clicked_resource_ids
    profile.top_tags = fresh.top_tags
    return profile