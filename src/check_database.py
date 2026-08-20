import sqlite3
from pathlib import Path


print("目前執行檔案：", __file__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "financial.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM companies")
print("Companies:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM filings")
print("Filings:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM chunks")
print("Chunks:", cursor.fetchone()[0])

cursor.execute(
    """
    SELECT
        companies.ticker,
        chunks.section,
        COUNT(*) AS chunk_count
    FROM chunks
    JOIN filings
        ON chunks.filing_id = filings.id
    JOIN companies
        ON filings.company_id = companies.id
    GROUP BY
        companies.ticker,
        chunks.section
    ORDER BY
        companies.ticker,
        chunk_count DESC
    """
)

rows = cursor.fetchall()

current_ticker = None

for ticker, section, count in rows:
    if ticker != current_ticker:
        print("\n" + "=" * 40)
        print(ticker)
        print("=" * 40)

        current_ticker = ticker

    print(
        f"{section:<10} {count:>4} chunks"
    )

cursor.execute(
    """
    SELECT
        companies.ticker,
        chunks.section,
        chunks.title,
        chunks.chunk_index,
        chunks.chunk_text
    FROM chunks
    JOIN filings
        ON chunks.filing_id = filings.id
    JOIN companies
        ON filings.company_id = companies.id
    WHERE chunks.id = ?
    """,
    (782,)
)

row = cursor.fetchone()

print(row)

print("\n前 3 個 chunks：")

cursor.execute("""
SELECT
    chunks.id,
    companies.ticker,
    filings.form_type,
    filings.filing_date,
    chunks.section,
    chunks.title,
    chunks.chunk_index,
    substr(chunks.chunk_text, 1, 120)
FROM chunks
JOIN filings
    ON chunks.filing_id = filings.id
JOIN companies
    ON filings.company_id = companies.id
LIMIT 3
""")

for row in cursor.fetchall():
    print("\n", row)
    
# %%
cursor.execute("""
SELECT
    section,
    COUNT(*)
FROM chunks
GROUP BY section
ORDER BY COUNT(*) DESC
""")

rows = cursor.fetchall()

print("\nSection 統計：")

for row in rows:
    print(row)


print("\nFilings：")

cursor.execute("""
SELECT
    filings.id,
    companies.ticker,
    filings.form_type,
    filings.filing_date,
    filings.source_file
FROM filings
JOIN companies
    ON filings.company_id = companies.id
ORDER BY filings.id
""")

for row in cursor.fetchall():
    print(row)

# %%
conn.close()