import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ChatRepository:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        self._lock = threading.Lock()
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
                """
            )
            self._connection.commit()

    def ensure_session(self, session_id: str | None, user_id: str) -> str:
        with self._lock:
            if session_id:
                existing = self._connection.execute(
                    "SELECT id FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if existing:
                    return session_id

            generated = session_id or str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions (id, user_id, created_at) VALUES (?, ?, ?)",
                (generated, user_id, now),
            )
            self._connection.commit()
            return generated

    def add_message(self, session_id: str, role: str, content: str) -> int:
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            cursor = self._connection.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            self._connection.commit()
            last_id = cursor.lastrowid
            if last_id is None:
                raise RuntimeError("Failed to persist message in SQLite")
            return int(last_id)

    def get_history(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = self._connection.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in rows][::-1]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
