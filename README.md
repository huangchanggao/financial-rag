# Financial RAG — SEC 10-K Analysis System

A multi-company, adaptive Retrieval-Augmented Generation (RAG) system for analyzing SEC 10-K filings.

使用多公司 SEC 10-K 財報建立的 Adaptive Retrieval-Augmented Generation（RAG）系統，結合結構化文件解析、LLM Section Selection、FAISS Retrieval、Cross-Encoder Reranking 與 Evidence Sufficiency Judging，根據問題動態決定是否需要進行第二輪 retrieval。

---

## Overview｜專案概述

Large SEC 10-K filings contain many sections and hundreds of text chunks. Searching the entire filing directly with vector similarity may retrieve semantically similar but contextually incorrect evidence.

SEC 10-K 文件篇幅很長，包含多個不同 Item 與大量文字 chunks。若直接對整份文件進行向量搜尋，容易抓到「文字看起來相似，但實際用途不同」的內容。

This project therefore uses an adaptive, section-aware retrieval pipeline.

本專案採用 adaptive、section-aware 的 retrieval 架構：

1. Identify the company from the user's question.  
   從問題中辨識公司。

2. Load the company's SEC 10-K section metadata from SQLite.  
   從 SQLite 取得該公司的 10-K section metadata。

3. Use an LLM to select the section most likely to contain the answer.  
   使用 LLM 判斷最可能包含答案的 section。

4. Perform FAISS retrieval only inside the selected company and section.  
   僅在指定公司與 section 中使用 FAISS 搜尋候選 chunks。

5. Rerank the retrieved candidates using a Cross-Encoder.  
   使用 Cross-Encoder 對候選 chunks 重新排序。

6. Use an Evidence Judge to determine whether the current evidence is sufficient.  
   使用 Evidence Judge 判斷目前取得的 evidence 是否足以回答問題。

7. If evidence is insufficient, dynamically choose a second retrieval strategy:
   - search deeper inside the same section
   - retrieve evidence from another SEC section

   若 evidence 不足，系統會動態決定：
   - 在同一個 section 中擴大搜尋
   - 或前往另一個 section 搜尋補充 evidence

8. Combine evidence from multiple retrieval rounds and rerank it again.  
   將多輪 retrieval 的 evidence 合併後再次 rerank。

9. Generate a grounded answer using only retrieved SEC filing evidence.  
   最後只根據 retrieval 找到的 SEC filing evidence 產生回答。

Current companies:

目前已加入：

- NVIDIA (`NVDA`)
- Advanced Micro Devices (`AMD`)
- Microsoft (`MSFT`)

---

# Architecture｜系統架構

```text
User Question
使用者問題
      |
      v
Company / Ticker Router
公司辨識
      |
      v
SQLite Section Metadata
取得公司 Section Metadata
      |
      v
LLM Section Selector
選擇最可能的 SEC Section
      |
      v
Round 1 Retrieval
FAISS within Ticker + Section
      |
      v
Cross-Encoder Reranking
候選 Chunk 重新排序
      |
      v
Top-K Evidence
      |
      v
Evidence Judge
      |
      +------------------------------+
      |              |               |
      v              v               v
    answer       same_section     new_section
      |              |               |
      |              v               v
      |        Search deeper      Search another
      |        in same section    SEC section
      |              |               |
      |              +-------+-------+
      |                      |
      |                      v
      |             Combine Evidence
      |                      |
      |                      v
      |             Final Reranking
      |                      |
      +-----------+----------+
                  |
                  v
         Grounded LLM Answer
         有來源依據的回答
```

---

## Adaptive Retrieval Decisions｜動態 Retrieval 決策

The Evidence Judge supports three retrieval actions.

Evidence Judge 會在三種 action 中選擇一種：

### `answer`

Current evidence is already sufficient.

目前 evidence 已足以回答問題：

```text
Round 1
  ↓
Evidence sufficient
  ↓
Answer
```

---

### `same_section`

The selected section is correct, but the current Top-K chunks do not provide enough coverage.

Section 選對了，但目前 Top-K chunks 不夠完整：

```text
Item 1A
  ↓
Top 5 evidence insufficient
  ↓
same_section
  ↓
Search Item 1A again
with a larger candidate pool
  ↓
Final reranking
```

The second query also incorporates the missing evidence identified by the Evidence Judge.

第二輪 query 會加入 Judge 判斷出的 missing evidence，使搜尋更有針對性。

Example:

```text
Question:
What are the main risk factors facing NVIDIA?

Round 1:
Item 1A

Judge:
same_section

Round 2:
Item 1A with expanded retrieval
```

---

### `new_section`

The current section contains useful evidence, but another SEC section is required to answer the full question.

目前 section 有部分 evidence，但完整回答需要另一個 section：

```text
Item 1C
  ↓
Evidence Judge
  ↓
Missing risk evidence
  ↓
new_section
  ↓
Item 1A
```

This is useful for questions that require evidence across multiple SEC sections.

適合需要跨 section 整合資訊的問題。

---

# Features｜主要功能

## 1. Multi-company SEC Ingestion｜多公司 SEC 財報處理

The ingestion pipeline supports multiple companies and stores company, filing, and chunk metadata in SQLite.

資料處理流程支援多家公司，並使用 SQLite 管理公司、財報與 chunk metadata。

Current companies:

```text
NVDA
AMD
MSFT
```

---

## 2. SEC 10-K Section Parsing｜10-K Section 解析

The parser detects standard SEC 10-K sections such as:

Parser 會辨識標準 10-K Item，例如：

```text
Item 1
Item 1A
Item 1B
Item 1C
Item 2
Item 3
Item 4
Item 5
Item 6
Item 7
Item 7A
Item 8
Item 9
Item 9A
Item 9B
Item 9C
Item 10
Item 11
Item 12
Item 13
Item 14
Item 15
Item 16
```

Section identifiers and section titles are stored separately.

Section 編號與標題分開保存，例如：

```text
section: Item 1A
title: RISK FACTORS
```

This keeps section identifiers stable while preserving semantic information from the original filing.

這樣可以保留穩定的 section metadata，同時保留原始 SEC 標題中的語意資訊。

---

## 3. Section-aware Chunking｜依 Section 切分 Chunks

Each extracted section is split into overlapping text chunks while preserving metadata.

每個 section 會切成具有 overlap 的文字 chunks，同時保留：

```text
ticker
filing date
form type
section
section title
chunk index
chunk id
```

Current text splitting settings:

```text
chunk size: 1000 characters
chunk overlap: 150 characters
```

---

## 4. SQLite Metadata Storage｜SQLite 資料管理

The project uses three main relational tables:

```text
companies
filings
chunks
```

Relationships:

```text
Company
   |
   v
Filing
   |
   v
Chunks
```

### companies

```text
id
ticker
company_name
cik
```

### filings

```text
id
company_id
form_type
filing_date
source_file
```

### chunks

```text
id
filing_id
section
title
chunk_index
chunk_text
```

This avoids repeatedly storing company and filing metadata inside every database row.

透過 relational schema 避免在每個 chunk 中重複保存完整 company / filing 資訊。

---

## 5. LLM Section Selection｜LLM Section 選擇

Instead of immediately searching every chunk in the filing, the system first provides the available SEC sections for the detected company to an LLM.

系統不會直接搜尋整份 filing，而是先把該公司的 SEC section metadata 提供給 LLM。

Example:

```text
Question:
What are AMD's main risk factors?

Selected section:
Item 1A
```

The selected section becomes a metadata filter for the first retrieval round.

---

## 6. FAISS Candidate Retrieval｜FAISS 候選檢索

After section selection, FAISS performs semantic retrieval only inside:

```text
selected ticker
+
selected section
```

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

FAISS is used as a candidate-generation stage rather than the final relevance decision.

FAISS 主要負責快速取得候選 chunks，不直接決定最後 evidence。

Typical first-round configuration:

```text
FAISS candidates: 15
Reranked evidence: Top 5
```

For a `same_section` retry, the candidate pool can be expanded to search deeper inside the selected section.

---

## 7. Cross-Encoder Reranking｜Cross-Encoder 重新排序

FAISS candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder evaluates the question and candidate chunk together:

```text
Question + Chunk
```

This provides a more detailed relevance estimate than vector similarity alone.

Cross-Encoder 會同時讀取 query 與 chunk，因此可進一步過濾只是在 embedding space 中相似、但實際不適合回答問題的內容。

---

## 8. Evidence Sufficiency Judge｜Evidence 足夠性判斷

After the first retrieval round, an LLM-based Evidence Judge evaluates whether the current Top-K chunks are sufficient.

第一輪 retrieval 後，Evidence Judge 會判斷目前 evidence 是否足夠。

Structured decision:

```json
{
  "sufficient": false,
  "missing_evidence": "Additional risk factors from the same Risk Factors section.",
  "retrieval_action": "same_section",
  "next_section": "Item 1A"
}
```

Possible actions:

```text
answer
same_section
new_section
```

The judge evaluates the retrieved evidence rather than assuming that selecting the correct section automatically means sufficient evidence has been retrieved.

這使系統能區分：

```text
Section 選對，但 chunks 不夠
```

與：

```text
需要其他 section 的 evidence
```

---

## 9. Multi-round Retrieval｜多輪 Retrieval

The system currently supports up to two retrieval rounds.

目前最多執行兩輪 retrieval。

### Same-section retry

```text
Round 1:
candidate_k = 15
Top 5

Round 2:
candidate_k = 30
Top 10
```

The second query incorporates the missing evidence identified by the Judge.

---

### New-section retrieval

If another section is required:

```text
Round 1 Section
      +
Round 2 Section
      ↓
Combined Evidence
      ↓
Cross-Encoder
      ↓
Final Top 6
```

Duplicate chunks are removed before the final reranking stage.

---

## 10. Grounded Answer Generation｜有依據的回答生成

The final LLM is instructed to answer using retrieved SEC filing evidence.

最終回答只應根據 retrieval 得到的 SEC filing context。

Rules include:

- Do not invent financial facts or numbers.
- Do not rely on unsupported outside information.
- State when retrieved evidence is insufficient.
- Include SEC filing source information.

Current LLM:

```text
openai/gpt-oss-20b
```

via Groq.

LLM construction is centralized in:

```text
src/llm_client.py
```

---

## 11. FastAPI Interface｜API 介面

The RAG pipeline is also exposed through FastAPI.

RAG 系統提供 FastAPI endpoint，可供未來 Web UI 或其他服務呼叫。

Start the API:

```bash
python -m uvicorn src.api:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Request:

```json
{
  "question": "What cybersecurity risks does Microsoft disclose?"
}
```

Example response structure:

```json
{
  "question": "What cybersecurity risks does Microsoft disclose?",
  "ticker": "MSFT",
  "selected_section": "Item 1C",
  "retrieval_rounds": 2,
  "additional_section": "Item 1A",
  "answer": "...",
  "sources": [
    {
      "ticker": "MSFT",
      "form_type": "10-K",
      "filing_date": "2026-07-29",
      "section": "Item 1C",
      "chunk_id": 1234
    }
  ]
}
```

---

# Data Pipeline｜資料處理流程

```text
SEC EDGAR
    |
    v
Download 10-K HTML
下載 SEC Filing
    |
    v
HTML Cleaning
HTML 清理
    |
    v
Section Detection
Section 辨識
    |
    v
Section-aware Chunking
依 Section 切 Chunk
    |
    v
SQLite
儲存結構化 Metadata
    |
    v
Embedding Generation
    |
    v
FAISS Vector Index
建立向量索引
```

---

# Project Structure｜專案結構

```text
financial-rag/
│
├── evaluation/
│   ├── evaluation.py
│   ├── multi_company_dataset.json
│   ├── adaptive_evaluation.py
│   └── adaptive_retrieval_eval.json
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── build_index.py
│   ├── check_database.py
│   ├── database.py
│   ├── download.py
│   ├── ingest.py
│   ├── llm_client.py
│   ├── rag.py
│   ├── section_router.py
│   └── test_section_parser.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Generated data is excluded from Git, including:

```text
.env
database/*.db
vectorstore/
data/
```

Therefore API credentials, downloaded SEC filings, generated SQLite databases, and FAISS indexes are not committed to the repository.

---

# Installation｜安裝

## 1. Clone the repository

```bash
git clone https://github.com/huangchanggao/financial-rag.git
cd financial-rag
```

---

## 2. Create a Conda environment

```bash
conda create -n financial-rag python=3.11
conda activate financial-rag
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a local `.env` file based on:

```text
.env.example
```

and provide the required API credentials.

Do not commit `.env` to Git.

---

# Running the Pipeline｜執行流程

## 1. Initialize the database｜建立資料庫

```bash
python src/database.py
```

---

## 2. Download a 10-K filing｜下載 SEC 10-K

```bash
python src/download.py
```

Enter the requested company information when prompted.

---

## 3. Ingest the filing｜解析並寫入 SQLite

```bash
python src/ingest.py
```

The filing passes through:

```text
HTML cleaning
→ Section detection
→ Section-aware chunking
→ SQLite storage
```

---

## 4. Build the FAISS index｜建立 FAISS Index

```bash
python src/build_index.py
```

---

## 5. Run the RAG system｜執行 RAG

Because the project uses package-style imports, run the RAG module from the repository root:

```bash
python -m src.rag
```

Example:

```text
What are the main risk factors facing NVIDIA?
```

---

## 6. Run the API｜啟動 API

```bash
python -m uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# Evaluation｜評估

The project includes both earlier retrieval evaluation experiments and a dedicated adaptive-retrieval evaluation.

專案包含基礎 retrieval evaluation，以及目前針對 adaptive retrieval architecture 建立的 evaluation。

Adaptive evaluation dataset:

```text
15 English questions
3 companies

9 single-section questions
6 cross-section questions
```

Companies:

```text
NVIDIA
AMD
Microsoft
```

Each question is manually labeled with the SEC section or sections expected to contain the required evidence.

---

## Adaptive Retrieval Evaluation

The current evaluation compares:

### Baseline

```text
LLM Section Selection
      ↓
One selected section
```

against:

### Adaptive

```text
LLM Section Selection
      ↓
First retrieval
      ↓
Evidence Judge
      ↓
answer / same_section / new_section
```

Primary metric:

```text
Expected Section Recall
```

This metric measures whether the retrieval process reaches the SEC sections manually labeled as necessary for the question.

It does **not** represent final-answer factual accuracy.

---

## Preliminary Results

Current evaluation results from one 15-question run:

| Metric | Baseline | Adaptive |
|---|---:|---:|
| Overall Expected Section Recall | 76.67% | **86.67%** |
| Single-section Recall | **100.00%** | **100.00%** |
| Cross-section Recall | 41.67% | **66.67%** |

Adaptive retrieval improved:

```text
Overall Expected Section Recall
76.67% → 86.67%
(+10.00 percentage points)
```

The largest improvement occurred on cross-section questions:

```text
41.67% → 66.67%
(+25.00 percentage points)
```

while single-section recall remained:

```text
100% → 100%
```

This suggests that adaptive retrieval improves section coverage on questions requiring evidence from multiple areas of a filing without reducing performance on simpler single-section questions.

---

## Retrieval Behavior

During the same evaluation run:

```text
Average Retrieval Rounds:      1.47

Second-round Trigger Rate:     46.67%

Direct Answer:                 53.33%
Same-section Retry:            26.67%
New-section Retrieval:         20.00%
```

This indicates that the system does not always execute an expensive second retrieval round.

Instead, the Evidence Judge conditionally decides whether additional retrieval is needed.

---

## Evaluation Limitations

The current evaluation is preliminary.

目前 evaluation 仍屬於初步實驗。

Important limitations include:

- The adaptive benchmark currently contains only 15 questions.
- The evaluation covers only three companies.
- Current results are based on a small number of runs.
- Expected Section Recall evaluates section coverage rather than final answer correctness.
- LLM section routing and evidence judging may show some run-to-run variation.

Observed failure modes include:

### Premature stopping

The Evidence Judge may occasionally decide that the first-round evidence is sufficient even when another section could provide evidence for another part of a multi-part question.

### Initial section routing errors

If the Section Selector chooses an incorrect first section, the Evidence Judge may sometimes retry the same incorrect section instead of recovering by selecting a better one.

These cases are retained as error-analysis examples rather than being removed from the evaluation set.

---

# Current Limitations｜目前限制

The current system still has several limitations:

- Company routing currently depends on supported company names and tickers.  
  公司辨識目前主要依賴已支援的 company names / tickers。

- The first-stage LLM Section Selector selects one primary section.  
  第一階段仍先選擇一個 primary section。

- Adaptive retrieval currently supports at most two rounds.  
  目前最多支援兩輪 retrieval。

- Evidence sufficiency decisions are LLM-based.  
  Evidence Judge 仍由 LLM 進行判斷。

- The system may stop retrieval too early for some multi-part questions.  
  部分 multi-part question 可能出現 premature stopping。

- An incorrect first section may not always be recovered successfully.  
  第一輪 routing 錯誤時，目前不一定能完全修正。

- SEC HTML structure differs across companies and filings.  
  不同公司的 SEC HTML 結構可能存在差異。

- Section parsing still contains heuristic rules.  
  Section parser 目前仍包含 heuristic 邏輯。

- The current dataset contains only a limited number of companies and filings.  
  目前資料規模仍有限。

- The current embedding model is primarily designed for English semantic retrieval.  
  目前 embedding model 主要針對英文語意檢索。

- No production frontend is currently included.  
  目前尚未建立正式 production frontend。

---

# Planned Improvements｜後續方向

Possible next steps include:

- Expand the adaptive evaluation dataset.  
  擴大 adaptive retrieval evaluation dataset。

- Run repeated evaluations and report mean / variance.  
  進行多次 evaluation，降低單次 LLM variation 的影響。

- Improve Evidence Judge stopping decisions.  
  改善 premature stopping。

- Improve recovery from incorrect first-section routing.  
  加強第一輪 section routing 錯誤後的 recovery。

- Evaluate chunk-level evidence quality in addition to section recall.  
  除了 section recall，也評估 chunk-level evidence quality。

- Add answer grounding / factuality evaluation.  
  加入 final answer grounding 與 factual correctness evaluation。

- Improve multilingual query handling.  
  改善中文等 multilingual query 的 retrieval。

- Support additional companies and filing years.  
  擴充公司與不同年份 filing。

- Support additional SEC filing types such as 10-Q.  
  未來支援 10-Q 等其他 SEC filing type。

- Evaluate retrieval latency and LLM token cost.  
  評估 retrieval latency 與 token usage。

- Build an interactive frontend.  
  建立 Web UI。

---

# Tech Stack｜技術

```text
Python
SQLite
FAISS
Sentence Transformers
Cross-Encoder
LangChain
Groq
FastAPI
Pydantic
BeautifulSoup
SEC EDGAR
```

Models:

```text
Embedding:
sentence-transformers/all-MiniLM-L6-v2

Reranker:
cross-encoder/ms-marco-MiniLM-L-6-v2

LLM:
openai/gpt-oss-20b
via Groq
```

---

# Project Goal｜專案目標

The goal of this project is not only to build a financial question-answering demo, but to explore how retrieval architecture affects evidence quality when processing long and structured financial documents.

本專案的目標不只是建立財報問答 Demo，而是實際研究：

```text
Document Structure
        +
Metadata Filtering
        +
Semantic Retrieval
        +
Reranking
        +
Adaptive Retrieval
        +
Evidence Sufficiency
        +
LLM Generation
```

如何共同影響長篇結構化金融文件的問答品質。

The project currently focuses on:

- Structured SEC document ingestion  
  SEC 結構化文件處理

- Metadata-aware retrieval  
  Metadata-aware retrieval

- Section-aware retrieval  
  Section-aware retrieval

- Semantic candidate generation  
  語意候選檢索

- Cross-Encoder reranking  
  Cross-Encoder 重新排序

- Adaptive multi-round retrieval  
  動態多輪 retrieval

- Evidence sufficiency judging  
  Evidence 足夠性判斷

- Grounded LLM generation  
  有 evidence 支持的 LLM 回答

- Retrieval evaluation and error analysis  
  Retrieval evaluation 與錯誤分析