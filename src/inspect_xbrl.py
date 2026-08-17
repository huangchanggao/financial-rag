from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent
FILINGS_DIR = BASE_DIR / "data" / "filings"


def main():
    html_files = list(FILINGS_DIR.glob("*.html"))

    if not html_files:
        print("找不到 HTML 財報")
        return

    filing_path = max(
        html_files,
        key=lambda path: path.stat().st_mtime
    )

    print("檢查檔案：", filing_path.name)

    html = filing_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    soup = BeautifulSoup(html, "html.parser")

    tag_names = []

    for tag in soup.find_all():
        if tag.name:
            tag_names.append(tag.name)

    counts = Counter(tag_names)

    print("\n=== 含有 ix / xbrl / gaap 的 tag ===")

    found = False

    for name, count in counts.most_common():
        lower = name.lower()

        if (
            "ix" in lower
            or "xbrl" in lower
            or "gaap" in lower
        ):
            print(f"{name}: {count}")
            found = True

    if not found:
        print("沒有找到符合條件的 tag")

    print("\n=== HTML 中是否直接存在關鍵字 ===")

    keywords = [
        "ix:header",
        "ix:hidden",
        "xbrli:",
        "us-gaap:",
        "iso4217:"
    ]

    html_lower = html.lower()

    for keyword in keywords:
        print(
            keyword,
            "=>",
            keyword.lower() in html_lower
        )


if __name__ == "__main__":
    main()