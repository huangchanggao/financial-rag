from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"


def main():
    txt_files = list(CLEAN_DIR.glob("*.txt"))

    if not txt_files:
        print("找不到清理後的 txt 財報")
        return

    latest_file = max(
        txt_files,
        key=lambda path: path.stat().st_mtime
    )

    print("檢查檔案：")
    print(latest_file.name)

    text = latest_file.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    print("\n=== 找到的 ITEM 標題候選 ===\n")

    pattern = re.compile(
        r"(?mi)^ITEM\s+\d+[A-Z]?\s*\.?.*$"
    )

    matches = pattern.findall(text)

    for i, match in enumerate(matches, start=1):
        print(f"{i:02d}. {match.strip()}")

    print("\n總共找到：", len(matches))


if __name__ == "__main__":
    main()