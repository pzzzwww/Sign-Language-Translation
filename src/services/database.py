"""
SQLite 数据库初始化与连接管理。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import ROOT

DB_PATH = ROOT / "data" / "mute.db"


def get_db() -> sqlite3.Connection:
    """获取数据库连接（每次调用返回新连接，线程安全）。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """初始化数据库表结构（幂等）。"""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS translation_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                create_time     TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                original_token  TEXT    NOT NULL,
                translated_text TEXT    NOT NULL,
                audio_path      TEXT    DEFAULT '',
                status          TEXT    NOT NULL DEFAULT 'pending',
                duration_sec    REAL    DEFAULT 0.0
            )
        """)
