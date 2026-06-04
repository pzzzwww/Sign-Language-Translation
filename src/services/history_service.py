"""
历史记录 CRUD 服务。
"""

from __future__ import annotations

from typing import Any

from src.services.database import get_db
from src.config import AUDIO_DIR


def create_record(tokens: list[str], text: str) -> int:
    """创建历史记录，返回新记录 ID。"""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO translation_history (original_token, translated_text, status) VALUES (?, ?, 'pending')",
            [" ".join(tokens), text],
        )
        return cur.lastrowid


def update_audio_path(record_id: int, filename: str, duration: float) -> None:
    """更新记录的音频文件路径和时长。"""
    with get_db() as conn:
        conn.execute(
            "UPDATE translation_history SET audio_path = ?, status = 'completed', duration_sec = ? WHERE id = ?",
            [filename, duration, record_id],
        )


def get_all_records() -> list[dict[str, Any]]:
    """获取所有历史记录（按时间倒序）。"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, create_time, original_token, translated_text, audio_path, status, duration_sec "
            "FROM translation_history ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_record(record_id: int) -> dict[str, Any] | None:
    """获取单条记录。"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, create_time, original_token, translated_text, audio_path, status, duration_sec "
            "FROM translation_history WHERE id = ?",
            [record_id],
        ).fetchone()
        return dict(row) if row else None


def delete_record(record_id: int) -> bool:
    """删除记录及对应的音频文件。返回是否删除成功。"""
    record = get_record(record_id)
    if not record:
        return False

    if record["audio_path"]:
        audio_file = AUDIO_DIR / record["audio_path"]
        try:
            audio_file.unlink(missing_ok=True)
        except OSError:
            pass

    with get_db() as conn:
        conn.execute("DELETE FROM translation_history WHERE id = ?", [record_id])

    return True
