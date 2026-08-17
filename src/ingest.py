# %%
from pathlib import Path

from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter

import sqlite3

import re

# %%

BASE_DIR = Path(__file__).resolve().parent.parent
FILINGS_DIR = BASE_DIR / "data" / "filings"
CLEAN_DIR = BASE_DIR / "data" / "clean"


def find_latest_filing():
    html_files = list(FILINGS_DIR.glob("*.html"))

    if not html_files:
        raise FileNotFoundError("data/filings 中找不到 HTML 財報")

    latest_file = max(
        html_files,
        key=lambda path: path.stat().st_mtime
    )

    return latest_file


def clean_html(html_path):
    print(f"正在處理：{html_path.name}")

    html = html_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    soup = BeautifulSoup(html, "html.parser")

    # 直接刪除整個 Inline XBRL header
    for tag in soup.find_all("ix:header"):
        tag.decompose()

    print(
        "清除後 ix:header 數量：",
        len(soup.find_all("ix:header"))
    )

    print(
        "清除後 xbrli:context 數量：",
        len(soup.find_all("xbrli:context"))
    )

    # 移除一般不需要的 HTML
    for tag in soup.find_all(
        ["script", "style", "noscript", "meta", "link"]
    ):
        tag.decompose()

    # 移除 display:none 的隱藏內容
    for tag in soup.find_all(style=True):
        style = tag.get("style", "").replace(" ", "").lower()

        if "display:none" in style:
            tag.decompose()

    # 正文中的 Inline XBRL 標籤只移除標籤本身，保留文字
    for tag in soup.find_all([
        "ix:nonfraction",
        "ix:nonnumeric",
        "ix:continuation"
    ]):
        tag.unwrap()

    text = soup.get_text(separator="\n")

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    clean_text = "\n".join(lines)

    return clean_text

def save_clean_text(text, original_path):
    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_name = original_path.stem + ".txt"
    output_path = CLEAN_DIR / output_name

    output_path.write_text(
        text,
        encoding="utf-8"
    )

    return output_path


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    return chunks

def split_into_sections(text):
    pattern = re.compile(
        r"(?mi)^Item[\s\xa0]+"
        r"(1A|1B|1C|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15|16)"
        r"\s*\.\s*(.*)$"
    )

    matches = list(pattern.finditer(text))

    # 每個 Item 的所有出現位置
    item_matches = {}

    for match in matches:
        item_number = match.group(1).upper()

        if item_number not in item_matches:
            item_matches[item_number] = []

        item_matches[item_number].append(match)

    # 取第二次出現，視為真正正文 section
    real_sections = []

    for item_number, occurrences in item_matches.items():
        if len(occurrences) >= 2:
            real_sections.append(
                occurrences[1]
            )

    # 按照文件中的位置排序
    real_sections.sort(
        key=lambda match: match.start()
    )

    sections = []

    for i, match in enumerate(real_sections):
        start = match.start()

        if i + 1 < len(real_sections):
            end = real_sections[i + 1].start()
        else:
            end = len(text)

        item_number = match.group(1).upper()
        title = match.group(2).strip()

        section_name = f"Item {item_number}"

        if title:
            section_name += f" - {title}"

        section_text = text[start:end]

        sections.append({
            "section": section_name,
            "text": section_text
        })

    return sections

def main():
    filing_path = find_latest_filing()

    clean_text = clean_html(filing_path)

    output_path = save_clean_text(
        clean_text,
        filing_path
    )

    sections = split_into_sections(clean_text)
    
    sectioned_chunks = []
    
    chunk_index = 0
    
    for section in sections:
        chunks = split_text(
            section["text"]
        )
    
        for chunk in chunks:
            sectioned_chunks.append({
                "chunk_index": chunk_index,
                "section": section["section"],
                "text": chunk
            })
    
            chunk_index += 1
    
    filing_date = filing_path.stem.split("_")[1]
    
    save_chunks_to_database(
        sectioned_chunks,
        filing_path.name,
        filing_date
    )

    print("\n清理完成")

# %%

DB_PATH = BASE_DIR / "database" / "financial.db"
# %%

def save_chunks_to_database(
    chunks,
    source_file,
    filing_date
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 先加入 NVIDIA 公司
    cursor.execute("""
    INSERT OR IGNORE INTO companies (
        ticker,
        company_name,
        cik
    )
    VALUES (?, ?, ?)
    """, (
        "NVDA",
        "NVIDIA Corporation",
        "0001045810"
    ))

    cursor.execute("""
    SELECT id
    FROM companies
    WHERE ticker = ?
    """, ("NVDA",))

    company_id = cursor.fetchone()[0]

    # 建立 filing
    cursor.execute("""
    INSERT INTO filings (
        company_id,
        form_type,
        filing_date,
        source_file
    )
    VALUES (?, ?, ?, ?)
    """, (
        company_id,
        "10-K",
        filing_date,
        source_file
    ))

    filing_id = cursor.lastrowid

    # 寫入 chunks
# 寫入 chunks
    for chunk in chunks:
    
        cursor.execute("""
        INSERT INTO chunks (
            filing_id,
            section,
            chunk_index,
            chunk_text
        )
        VALUES (?, ?, ?, ?)
        """, (
            filing_id,
            chunk["section"],
            chunk["chunk_index"],
            chunk["text"]
        ))

    conn.commit()
    conn.close()

    print(
        f"\n已寫入 {len(chunks)} 個 chunks 到 SQLite"
    )
# %%

if __name__ == "__main__":
    main()