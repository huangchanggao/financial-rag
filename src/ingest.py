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

def chunk_sections(sections):
    all_chunks = []
    chunk_index = 0

    for section in sections:
        section_name = section["section"]
        section_title = section.get("title", "")
        section_text = section["text"]

        text_chunks = split_text(section_text)

        for text_chunk in text_chunks:
            all_chunks.append({
                "section": section_name,
                "title": section_title,
                "chunk_index": chunk_index,
                "text": text_chunk
            })

            chunk_index += 1

    return all_chunks

ITEM_ORDER = [
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "9C",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16"
]


def find_body_item_matches(matches, text_length):
    # 找出所有 Item 1
    item1_matches = [
        match
        for match in matches
        if match.group(1).upper() == "1"
    ]

    # 如果 Item 1 出現至少兩次，
    # 第一個通常是 TOC，第二個視為正文起點
    if len(item1_matches) >= 2:
        body_start = item1_matches[1].start()

    elif len(item1_matches) == 1:
        body_start = item1_matches[0].start()

    else:
        body_start = 0

    # 只保留正文起點之後的 heading
    candidates = [
        match
        for match in matches
        if match.start() >= body_start
    ]

    selected = []
    last_position = body_start - 1

    # 按照正常 SEC Item 順序尋找
    for expected_item in ITEM_ORDER:

        possible_matches = [
            match
            for match in candidates
            if match.group(1).upper() == expected_item
            and match.start() > last_position
        ]

        if not possible_matches:
            continue

        chosen = possible_matches[0]

        selected.append(chosen)
        last_position = chosen.start()

    return selected

def extract_item_title(text, match):
    first_part = match.group(2).strip()

    lookahead = text[
        match.end():
        match.end() + 300
    ]

    lines = [
        line.strip()
        for line in lookahead.splitlines()
        if line.strip()
    ]

    parts = []

    if first_part:
        parts.append(first_part)

    for line in lines:
        if re.match(
            r"(?i)^Item\s+\d{1,2}[A-Z]?\s*\.",
            line
        ):
            break

        if len(line) > 120:
            break

        if not line.isupper():
            break

        parts.append(line)

        if len(parts) >= 3:
            break

    if not parts:
        return ""

    title = parts[0]

    for part in parts[1:]:
        # 如果前一段最後一個 token 很短，
        # 有可能是單字被 HTML 換行切開
        last_word = title.split()[-1]

        if (
            last_word.isalpha()
            and len(last_word) <= 12
            and part
            and part[0].isalpha()
        ):
            title = title + part
        else:
            title = title + " " + part

    return title.strip()

def split_into_sections(text):
    pattern = re.compile(
        r"(?mi)^Item[\s\xa0]+"
        r"(1A|1B|1C|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15|16)"
        r"\s*\.\s*(.*)$"
    )

    matches = list(pattern.finditer(text))

    real_sections = find_body_item_matches(
        matches,
        len(text)
    )

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

        title = extract_item_title(
            text,
            match
        )

        section_name = f"Item {item_number}"

        section_text = text[start:end]

        sections.append({
            "section": section_name,
            "title": title,
            "text": section_text
        })

    return sections

'''
    # 這段被註解掉，因為改了找section的方法
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
'''

# %%

DB_PATH = BASE_DIR / "database" / "financial.db"
# %%

def save_chunks_to_database(
    ticker,
    company_name,
    cik,
    filing_date,
    source_file,
    chunks
):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    # 建立公司
    cursor.execute(
        """
        INSERT OR IGNORE INTO companies (
            ticker,
            company_name,
            cik
        )
        VALUES (?, ?, ?)
        """,
        (
            ticker,
            company_name,
            cik
        )
    )

    # 取得 company_id
    cursor.execute(
        """
        SELECT id
        FROM companies
        WHERE cik = ?
        """,
        (cik,)
    )

    company_id = cursor.fetchone()[0]

    # 檢查 filing 是否已存在
    cursor.execute(
        """
        SELECT id
        FROM filings
        WHERE company_id = ?
          AND form_type = ?
          AND filing_date = ?
        """,
        (
            company_id,
            "10-K",
            filing_date
        )
    )

    existing_filing = cursor.fetchone()

    if existing_filing:
        print(
            f"Filing 已存在，跳過 ingest："
            f"{ticker} 10-K {filing_date}"
        )

        conn.close()
        return

    # 建立新的 filing
    cursor.execute(
        """
        INSERT INTO filings (
            company_id,
            form_type,
            filing_date,
            source_file
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            company_id,
            "10-K",
            filing_date,
            source_file
        )
    )

    filing_id = cursor.lastrowid

    # 寫入 chunks
    for chunk in chunks:
        cursor.execute(
            """
            INSERT INTO chunks (
                filing_id,
                section,
                title,
                chunk_index,
                chunk_text
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filing_id,
                chunk["section"],
                chunk["title"],
                chunk["chunk_index"],
                chunk["text"]
            )
        )

    conn.commit()
    conn.close()

    print(
        f"\n已寫入 {len(chunks)} 個 chunks 到 SQLite"
    )

# %%

def main():
    ticker = input(
        "Ticker: "
    ).strip().upper()

    company_name = input(
        "Company name: "
    ).strip()

    cik = input(
        "CIK: "
    ).strip().zfill(10)

    filing_file = input(
        "Filing filename: "
    ).strip()

    filing_path = FILINGS_DIR / filing_file

    if not filing_path.exists():
        print(
            f"找不到 filing：{filing_path}"
        )
        return

    parts = filing_file.split("_")

    if len(parts) < 3:
        print("檔名格式不正確")
        return

    filing_date = parts[1]

    print(
        f"\n開始處理：{filing_file}"
    )

    clean_text = clean_html(
        filing_path
    )

    sections = split_into_sections(
        clean_text
    )

    print(
        f"找到 {len(sections)} 個 sections"
    )

    chunks = chunk_sections(
        sections
    )

    print(
        f"產生 {len(chunks)} 個 chunks"
    )

    save_chunks_to_database(
        ticker=ticker,
        company_name=company_name,
        cik=cik,
        filing_date=filing_date,
        source_file=filing_file,
        chunks=chunks
    )
if __name__ == "__main__":
    main()