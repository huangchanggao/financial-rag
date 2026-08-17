# %%

import requests
import json
from pathlib import Path

# %%
HEADERS = {
    "User-Agent": "financial-rag-project huangchanggao1999@gmail.com"
}

CIK = "0001045810"   # NVIDIA
TICKER = "NVDA"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "filings"
# %%

def get_latest_10k():
    url = f"https://data.sec.gov/submissions/CIK{CIK}.json"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()

    recent = data["filings"]["recent"]

    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession_number = recent["accessionNumber"][i]
            primary_document = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]

            return accession_number, primary_document, filing_date

    return None


def download_10k():
    result = get_latest_10k()

    if result is None:
        print("找不到 10-K")
        return

    accession_number, primary_document, filing_date = result

    accession_no_dash = accession_number.replace("-", "")

    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(CIK)}/{accession_no_dash}/{primary_document}"
    )

    print("找到 NVIDIA 10-K")
    print("Filing date:", filing_date)
    print("URL:", filing_url)

    response = requests.get(filing_url, headers=HEADERS)
    response.raise_for_status()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / f"{TICKER}_{filing_date}_10K.html"

    output_path.write_text(
        response.text,
        encoding="utf-8"
    )

    print("下載完成：")
    print(output_path)
# %%



if __name__ == "__main__":
    download_10k()