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
                    user_name TEXT NOT NULL DEFAULT 'User',
                    title TEXT NOT NULL DEFAULT 'Novo chat',
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
            self._ensure_column(cursor, "sessions", "user_name", "TEXT NOT NULL DEFAULT 'User'")
            self._ensure_column(cursor, "sessions", "title", "TEXT NOT NULL DEFAULT 'Novo chat'")
            self._connection.commit()

    def _ensure_column(self, cursor: sqlite3.Cursor, table_name: str, column_name: str, definition: str) -> None:
        existing_columns = {
            row[1]
            for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    @staticmethod
    def _build_user_id(user_name: str) -> str:
        slug = "-".join(user_name.strip().lower().split())
        return slug or "user"

    def create_session(self, user_name: str) -> dict:
        clean_user_name = user_name.strip()
        if not clean_user_name:
            raise ValueError("user_name cannot be empty")

        session_id = str(uuid.uuid4())
        user_id = self._build_user_id(clean_user_name)
        title = f"Chat de {clean_user_name}"
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._connection.execute(
                "INSERT INTO sessions (id, user_id, user_name, title, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, clean_user_name, title, now),
            )
            self._connection.commit()

        return {
            "id": session_id,
            "user_id": user_id,
            "user_name": clean_user_name,
            "title": title,
            "created_at": now,
            "last_message_at": None,
        }

    def ensure_session(self, session_id: str | None, user_id: str, user_name: str) -> str:
        with self._lock:
            if session_id:
                existing = self._connection.execute(
                    "SELECT id FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if existing:
                    self._connection.execute(
                        "UPDATE sessions SET user_id = ?, user_name = ?, title = ? WHERE id = ?",
                        (user_id, user_name, f"Chat de {user_name}", session_id),
                    )
                    self._connection.commit()
                    return session_id

            generated = session_id or str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions (id, user_id, user_name, title, created_at) VALUES (?, ?, ?, ?, ?)",
                (generated, user_id, user_name, f"Chat de {user_name}", now),
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

    def list_sessions(self, limit: int = 100) -> list[dict]:
        rows = self._connection.execute(
            """
            SELECT
                s.id,
                s.user_id,
                s.user_name,
                s.title,
                s.created_at,
                MAX(m.created_at) AS last_message_at
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id, s.user_id, s.user_name, s.title, s.created_at
            ORDER BY COALESCE(MAX(m.created_at), s.created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
