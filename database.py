import sqlite3
from contextlib import contextmanager

DB_FILE = "cognigraph.db"

@contextmanager
def get_db_connection():
    """Context manager for database connection."""
    conn = sqlite3.connect(DB_FILE)
    try:
        yield conn
    finally:
        conn.close()

def initialize_database():
    """Initializes the database and creates the conversations table if it doesn't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                history TEXT NOT NULL
            )
        """)
        conn.commit()

def save_conversation(session_id, history):
    """Saves or updates a conversation in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, history)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET history = excluded.history
        """, (session_id, history))
        conn.commit()

def load_conversation(session_id):
    """Loads a conversation from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM conversations WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None

# Initialize the database when this module is loaded
initialize_database()
