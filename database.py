import sqlite3
import json

DB_NAME = "faces.db"


def connect():
    return sqlite3.connect(DB_NAME)


def create_table():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embeddings TEXT NOT NULL  -- JSON list of embedding lists
        )
    """)

    conn.commit()
    conn.close()


def insert_user(name, embeddings):
    """
    embeddings is a list of embedding lists (each embedding is a list of 512 floats)
    """
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("SELECT embeddings FROM users WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row:
        emb_list = json.loads(row[0])
        emb_list.extend(embeddings)  # append new scans
        cursor.execute(
            "UPDATE users SET embeddings = ? WHERE name = ?",
            (json.dumps(emb_list), name)
        )
    else:
        cursor.execute(
            "INSERT INTO users (name, embeddings) VALUES (?, ?)",
            (name, json.dumps(embeddings))
        )

    conn.commit()
    conn.close()


def get_all_users():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("SELECT name, embeddings FROM users")
    rows = cursor.fetchall()
    conn.close()

    users = []
    for name, emb_json in rows:
        emb_list = json.loads(emb_json)
        users.append((name, emb_list))

    return users