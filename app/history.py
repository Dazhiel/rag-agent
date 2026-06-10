"""
Chat history persistence backed by MySQL.
"""
import json
import re
from datetime import datetime
from typing import Any, List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class MySQLChatHistory(BaseChatMessageHistory):
    """Store one chat session in MySQL, isolated by session_id."""

    def __init__(self, session_id: str, config: Any):
        self.session_id = session_id
        self.config = config
        self._ensure_tables()

    def _connect(self):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency pymysql. Install it with: pip install pymysql"
            ) from exc

        kwargs = {
            "host": self.config.mysql_host,
            "port": self.config.mysql_port,
            "user": self.config.mysql_user,
            "password": self.config.mysql_password,
            "charset": self.config.mysql_charset,
            "autocommit": False,
            "cursorclass": pymysql.cursors.DictCursor,
        }

        try:
            return pymysql.connect(database=self.config.mysql_database, **kwargs)
        except pymysql.err.OperationalError as exc:
            if not exc.args or exc.args[0] != 1049:
                raise
            if not re.match(r"^[A-Za-z0-9_]+$", self.config.mysql_database):
                raise RuntimeError(
                    "MYSQL_DATABASE can only contain letters, numbers, and underscores"
                ) from exc

            with pymysql.connect(**kwargs) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{self.config.mysql_database}` "
                        "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                conn.commit()

            return pymysql.connect(database=self.config.mysql_database, **kwargs)

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(64) PRIMARY KEY,
                        title VARCHAR(255) NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        message_index INT UNSIGNED NOT NULL,
                        role VARCHAR(32) NOT NULL,
                        content LONGTEXT NULL,
                        message_json JSON NOT NULL,
                        created_at DATETIME NOT NULL,
                        UNIQUE KEY uk_session_message_index (session_id, message_index),
                        KEY idx_session_id (session_id),
                        CONSTRAINT fk_chat_messages_session
                            FOREIGN KEY (session_id)
                            REFERENCES chat_sessions(session_id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            conn.commit()

    def _ensure_session(self, conn) -> None:
        now = datetime.now()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_sessions (session_id, created_at, updated_at)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE updated_at = VALUES(updated_at)
                """,
                (self.session_id, now, now),
            )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return

        existing_count = len(self.messages)
        now = datetime.now()
        rows = []

        for offset, message in enumerate(messages):
            message_data = message_to_dict(message)
            payload = json.dumps(message_data, ensure_ascii=False)
            role = message_data.get("type", message.__class__.__name__)
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            rows.append(
                (
                    self.session_id,
                    existing_count + offset,
                    role,
                    content,
                    payload,
                    now,
                )
            )

        with self._connect() as conn:
            self._ensure_session(conn)
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chat_messages
                        (session_id, message_index, role, content, message_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = %s
                    WHERE session_id = %s
                    """,
                    (now, self.session_id),
                )
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT message_json
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY message_index ASC
                    """,
                    (self.session_id,),
                )
                rows = cursor.fetchall()

        payloads = []
        for row in rows:
            value = row["message_json"]
            payloads.append(json.loads(value) if isinstance(value, str) else value)

        return messages_from_dict(payloads)

    def clear(self) -> None:
        with self._connect() as conn:
            self._ensure_session(conn)
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM chat_messages WHERE session_id = %s",
                    (self.session_id,),
                )
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = %s
                    WHERE session_id = %s
                    """,
                    (datetime.now(), self.session_id),
                )
            conn.commit()


class HistoryManager:
    """Create chat history handles for different sessions."""

    def __init__(self, config: Any):
        self.config = config

    def get(self, session_id: str) -> MySQLChatHistory:
        return MySQLChatHistory(session_id, self.config)
