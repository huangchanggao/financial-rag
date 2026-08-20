from pathlib import Path

from ingest import (
    clean_html,
    split_into_sections
)


BASE_DIR = Path(__file__).resolve().parent.parent
FILINGS_DIR = BASE_DIR / "data" / "filings"


def main():
    files = list(
        FILINGS_DIR.glob("*.html")
    )

    if not files:
        print("找不到任何 HTML filing")
        return

    for filing_path in files:
        print("\n" + "=" * 80)
        print(f"Testing: {filing_path.name}")
        print("=" * 80)

        clean_text = clean_html(
            filing_path
        )

        sections = split_into_sections(
            clean_text
        )
        ###
        import re
        
        pattern = re.compile(
            r"(?mi)^Item[\s\xa0]+"
            r"(1A|1B|1C|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15|16)"
            r"\s*\.\s*(.*)$"
        )
        
        matches = list(pattern.finditer(clean_text))
        
        print("\nItem 1 candidates:")
        
        for match in matches:
            if match.group(1).upper() == "1":
                position = match.start()
                percentage = position / len(clean_text) * 100
        
                print(
                    f"position={position}, "
                    f"percentage={percentage:.2f}%, "
                    f"title={match.group(2).strip()}"
                )
        ###
        print(f"\n共找到 {len(sections)} 個 sections\n")

        for i, section in enumerate(
            sections,
            start=1
        ):
            print(
                f"{i:02d}. "
                f"{section['section']}"
            )


if __name__ == "__main__":
    main()