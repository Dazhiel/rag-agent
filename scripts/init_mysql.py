"""初始化用于保存历史会话的 MySQL 数据库。

运行方式：
    python scripts/init_mysql.py
"""
import re

import pymysql

from app.config import RAGConfig


def validate_database_name(name: str) -> None:
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise ValueError("MYSQL_DATABASE 只能包含英文字母、数字和下划线。")


def main() -> None:
    config = RAGConfig()
    validate_database_name(config.mysql_database)

    base_kwargs = {
        "host": config.mysql_host,
        "port": config.mysql_port,
        "user": config.mysql_user,
        "password": config.mysql_password,
        "charset": config.mysql_charset,
        "autocommit": False,
        "cursorclass": pymysql.cursors.DictCursor,
    }

    with pymysql.connect(**base_kwargs) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.mysql_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()

    with pymysql.connect(database=config.mysql_database, **base_kwargs) as conn:
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

    print(f"MySQL 数据库已初始化：{config.mysql_database}")


if __name__ == "__main__":
    main()
