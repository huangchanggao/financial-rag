import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "financial.db"


def get_connection():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(DB_PATH)

    return conn


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT UNIQUE NOT NULL,
        company_name TEXT NOT NULL,
        cik TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS filings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        form_type TEXT NOT NULL,
        filing_date TEXT,
        source_file TEXT,
        source_url TEXT,

        FOREIGN KEY (company_id)
        REFERENCES companies(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filing_id INTEGER NOT NULL,
        section TEXT,
        title TEXT,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
    
        FOREIGN KEY (filing_id)
        REFERENCES filings(id)
    )
    """)

    conn.commit()
    conn.close()

    print("Database tables created successfully.")


if __name__ == "__main__":
    create_tables()