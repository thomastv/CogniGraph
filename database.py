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
    """Initializes the database and creates tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Table for user preferences (key-value store)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()

def save_preference(key, value):
    """Saves or updates a user preference in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))
        conn.commit()

def load_all_preferences():
    """Loads all user preferences from the database into a dictionary."""
    prefs = {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM user_preferences")
        for row in cursor.fetchall():
            prefs[row[0]] = row[1]
    return prefs

# Initialize the database when this module is loaded
initialize_database()
