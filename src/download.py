# %%

import requests
from pathlib import Path

# %%
HEADERS = {
    "User-Agent": "financial-rag-project huangchanggao1999@gmail.com"
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "filings"

# %%


def get_latest_10k(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    recent = data["filings"]["recent"]

    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession_number = (
                recent["accessionNumber"][i]
            )

            primary_document = (
                recent["primaryDocument"][i]
            )

            filing_date = (
                recent["filingDate"][i]
            )

            return (
                accession_number,
                primary_document,
                filing_date
            )

    return None


# %%


def download_10k(ticker, cik):
    result = get_latest_10k(cik)

    if result is None:
        print(f"找不到 {ticker} 的 10-K")
        return

    (
        accession_number,
        primary_document,
        filing_date
    ) = result

    accession_no_dash = (
        accession_number.replace("-", "")
    )

    filing_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/"
        f"{accession_no_dash}/"
        f"{primary_document}"
    )

    print(f"找到 {ticker} 10-K")
    print("Filing date:", filing_date)
    print("URL:", filing_url)

    response = requests.get(
        filing_url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        DATA_DIR
        / f"{ticker}_{filing_date}_10K.html"
    )

    output_path.write_text(
        response.text,
        encoding="utf-8"
    )

    print("下載完成：")
    print(output_path)

    return output_path


# %%


def main():
    ticker = input(
        "請輸入股票代號，例如 NVDA、AMD："
    ).strip().upper()

    cik = input(
        "請輸入 10 位 CIK："
    ).strip()

    cik = cik.zfill(10)

    download_10k(
        ticker,
        cik
    )


# %%


if __name__ == "__main__":
    main()