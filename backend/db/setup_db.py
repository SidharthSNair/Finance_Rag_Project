import os
from sqlalchemy import create_engine, text

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "finance.db"))

def init_db():
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

    with engine.begin() as conn:
        # Companies table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY,
            ticker TEXT UNIQUE,
            name TEXT,
            sector TEXT
        );
        """))

        # Simple daily prices table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            trade_date TEXT, -- ISO date string
            close REAL
        );
        """))

        # Seed (safe upserts)
        conn.execute(text("""
        INSERT OR IGNORE INTO companies (id, ticker, name, sector) VALUES
          (1, 'AAPL', 'Apple Inc.', 'Technology'),
          (2, 'MSFT', 'Microsoft Corp.', 'Technology'),
          (3, 'GOOGL', 'Alphabet Inc.', 'Communication Services');
        """))

        conn.execute(text("""
        INSERT OR IGNORE INTO prices (id, ticker, trade_date, close) VALUES
          (1, 'AAPL', '2024-10-01', 175.2),
          (2, 'AAPL', '2024-10-02', 176.0),
          (3, 'MSFT', '2024-10-01', 405.5),
          (4, 'MSFT', '2024-10-02', 407.9),
          (5, 'GOOGL', '2024-10-01', 141.7),
          (6, 'GOOGL', '2024-10-02', 142.3);
        """))

    print(f"SQLite initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
