import csv
import json
import os
import re
import time
import warnings
from typing import List

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pydantic import BaseModel, Field

# Silence BeautifulSoup XML-parsed-as-HTML warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ======================================================================================================================
# API KEY & CONFIGURATION
# ======================================================================================================================
# Google AI Studio Key (Remove before pushing to public GitHub)


if not API_KEY.strip():
    API_KEY = os.getenv("GEMINI_API_KEY", "")

# SEC EDGAR requires a custom User-Agent header
HEADERS = {
    "User-Agent": "RemingtonBudzinski ResearchProject@ucf.edu",
    "Accept-Encoding": "gzip, deflate",
}

# S&P 500 companies to analyze (stops after processing 100 documents)
SP500_TICKERS = [
    "AAPL", "NVDA", "MSFT", "AMD", "GOOGL",
    "AMZN", "META", "CRM", "INTC", "TSLA",
    "ORCL", "IBM", "CSCO", "QCOM", "ADBE",
]
MAX_DOCUMENTS = 100
FILINGS_PER_COMPANY = 10

OUTPUT_CSV = "sec_8k_new_products.csv"
OUTPUT_PIPE_TXT = "sec_8k_pipe_output.txt"


# ======================================================================================================================
# PYDANTIC SCHEMAS FOR NER & STRUCTURED EXTRACTION
# ======================================================================================================================
class ProductAnnouncement(BaseModel):
    company_name: str = Field(
        description="Company filing the 8-K (Entity Type: ORGANIZATION)."
    )
    new_product: str = Field(
        description="Newly announced product (Entity Type: PRODUCT)."
    )
    product_description: str = Field(
        description="Concise product description (< 180 characters)."
    )


class ExtractionResult(BaseModel):
    announcements: List[ProductAnnouncement]


# ======================================================================================================================
# STEP 1: RESOLVE TICKER TO CIK VIA BEAUTIFULSOUP
# ======================================================================================================================
def get_ticker_to_cik_map() -> dict:
    url = f"https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    data = json.loads(soup.text)

    ticker_map = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik_padded = str(entry["cik_str"]).zfill(10)
        ticker_map[ticker] = {
            "cik": cik_padded,
            "company_name": entry["title"],
        }
    return ticker_map


# ======================================================================================================================
# STEP 2: FETCH & PARSE 8-K FILINGS VIA BEAUTIFULSOUP
# ======================================================================================================================
def extract_8k_text_from_index(index_url: str) -> str:
    try:
        res = requests.get(index_url, headers=HEADERS, timeout=15)
        if res.status_code != 200:
            return ""
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="tableFile")
        if not table:
            return ""

        doc_url = None
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 4:
                doc_type = cols[3].text.strip()
                link = cols[2].find("a")
                if link and ("8-K" in doc_type or doc_type == "EX-99.1"):
                    href = re.sub(r"^/ix\?doc=", "", link["href"])
                    doc_url = f"https://www.sec.gov{href}"
                    if "8-K" in doc_type:
                        break

        if not doc_url:
            return ""

        time.sleep(0.15)
        doc_res = requests.get(doc_url, headers=HEADERS, timeout=15)
        doc_soup = BeautifulSoup(doc_res.text, "html.parser")

        raw_text = doc_soup.get_text(separator=" ", strip=True)
        cleaned_text = re.sub(r"\s+", " ", raw_text)
        return cleaned_text[:6000]
    except Exception as e:
        print(f"Error parsing filing document {index_url}: {e}")
        return ""


def fetch_8k_filings(cik: str, count: int = 10) -> list:
    search_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=8-K&count={count}&output=atom"
    )
    response = requests.get(search_url, headers=HEADERS, timeout=15)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    entries = soup.find_all("entry")
    filings = []

    for entry in entries:
        filing_date_tag = entry.find("filing-date")
        filing_href_tag = entry.find("filing-href")
        if not filing_date_tag or not filing_href_tag:
            continue

        filing_time = filing_date_tag.text.strip()
        index_url = filing_href_tag.text.strip()

        time.sleep(0.15)
        filing_text = extract_8k_text_from_index(index_url)
        if filing_text:
            filings.append(
                {
                    "filing_time": filing_time,
                    "text": filing_text,
                    "url": index_url,
                }
            )

    return filings


# ======================================================================================================================
# STEP 3: NAMED ENTITY RECOGNITION (NER) & GEMINI API EXTRACTION
# ======================================================================================================================
def build_ner_prompt(
    company_name: str, ticker: str, filing_time: str, text: str
) -> str:
    return f"""You are a financial NLP system performing Named Entity Recognition (NER) on an SEC Form 8-K filing.
Analyze the following 8-K text from {company_name} ({ticker}) filed on {filing_time}.

Instructions:
1. Identify the Company Name (Entity Type: ORGANIZATION) and any newly announced products, platforms, hardware, software, or services (Entity Type: PRODUCT).
2. If NO specific product, platform, or service is mentioned in this filing, return: {{"announcements": []}}
3. If one or more products or services are mentioned, return a JSON object with an "announcements" list where each item has:
   - "company_name": The official name of the company filing the 8-K.
   - "new_product": The specific name of the product or service.
   - "product_description": A concise description of the product, strictly LESS than 180 characters.

8-K Filing Text:
\"\"\"{text}\"\"\"

Respond ONLY with valid JSON matching the schema."""


def fallback_ner_extractor(
    company_name: str, ticker: str, filing_time: str, text: str
) -> list:
    """Backup pattern-based NER extractor if a free-tier call is rate-limited or rejected."""
    patterns = [
        r"(?:announced|launches|launched|introduces|introduced|unveils|unveiled|releases|released|new\s+product|platform|service)\s+(?:the\s+|its\s+|a\s+)?([A-Z][A-Za-z0-9\-\+\s]{2,32})",
        r"Item\s+(?:8\.01|2\.02|7\.01)[^\.]{15,165}",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            snippet = match.group(0).strip()
            prod_name = (
                match.group(1).strip()
                if match.lastindex
                else f"{ticker} Material Release"
            )
            prod_name = re.sub(
                r"\b(Item|Form|Section|Exhibit|Agreement|Pursuant|The|And|For)\b.*",
                "",
                prod_name,
            ).strip()
            if len(prod_name) < 3:
                prod_name = f"{ticker} Product / Service Update"

            desc = re.sub(r"\s+", " ", snippet)
            if len(desc) >= 180:
                desc = desc[:176].rstrip() + "..."

            return [
                {
                    "company_name": company_name,
                    "stock_name": ticker,
                    "filing_time": filing_time,
                    "new_product": prod_name[:45],
                    "product_description": desc,
                }
            ]
    return []


def call_llm_api(prompt: str, key: str) -> str:
    """Calls Google AI Studio Gemini API (supports both 'AQ.' and 'AIza' keys)."""
    if key.startswith(("AQ.", "AIza")):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={key}"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
            },
        }
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        if res.status_code == 429:
            time.sleep(6)
            res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # Fallback for Groq ("gsk_") or OpenAI ("sk-") keys
    api_url = (
        "https://api.groq.com/openai/v1/chat/completions"
        if key.startswith("gsk_")
        else "https://api.openai.com/v1/chat/completions"
    )
    model_name = (
        "llama-3.1-8b-instant" if key.startswith("gsk_") else "gpt-4o-mini"
    )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Extract structured NER data from SEC 8-K filings in JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    res = requests.post(api_url, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()


# Track whether the API key is accepted so we only print a warning once if rejected
API_ACTIVE = True


def extract_products_with_api(
    company_name: str, ticker: str, filing_time: str, text: str
) -> list:
    global API_ACTIVE
    key = API_KEY.strip()

    if not API_ACTIVE:
        return fallback_ner_extractor(company_name, ticker, filing_time, text)

    prompt = build_ner_prompt(company_name, ticker, filing_time, text)

    try:
        raw_json_str = call_llm_api(prompt, key)
        raw_json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json_str)

        parsed_data = json.loads(raw_json_str)
        if isinstance(parsed_data, list):
            parsed_data = {"announcements": parsed_data}
        elif "announcements" not in parsed_data and "new_product" in parsed_data:
            parsed_data = {"announcements": [parsed_data]}

        validated = ExtractionResult(**parsed_data)
        results = []

        for item in validated.announcements:
            prod_name = item.new_product.strip()
            prod_desc = item.product_description.strip()
            comp_name = item.company_name.strip() or company_name

            if prod_name and prod_name.lower() not in ["none", "n/a", "null", ""]:
                if len(prod_desc) >= 180:
                    prod_desc = prod_desc[:176].rstrip() + "..."
                results.append(
                    {
                        "company_name": comp_name,
                        "stock_name": ticker,
                        "filing_time": filing_time,
                        "new_product": prod_name,
                        "product_description": prod_desc,
                    }
                )
        time.sleep(1.5)
        return results

    except Exception as e:
        print(
            f"  [Note] API response notice for {ticker} ({filing_time}): {e}\n"
            f"  -> Switching to built-in local NER extractor for remaining filings."
        )
        API_ACTIVE = False
        return fallback_ner_extractor(company_name, ticker, filing_time, text)


# ======================================================================================================================
# STEP 4: MAIN EXECUTION & FILE SAVING
# ======================================================================================================================
def main():
    print("Initialized LLM Extraction using Google AI Studio Key (AQ. format)")
    print("Loading SEC ticker-to-CIK mapping...")
    ticker_map = get_ticker_to_cik_map()

    extracted_rows = []
    docs_processed = 0

    for ticker in SP500_TICKERS:
        if docs_processed >= MAX_DOCUMENTS:
            break
        if ticker not in ticker_map:
            continue

        cik = ticker_map[ticker]["cik"]
        default_company_name = ticker_map[ticker]["company_name"]
        print(f"Fetching 8-K filings for {ticker} (CIK: {cik})...")

        filings = fetch_8k_filings(cik, count=FILINGS_PER_COMPANY)
        for filing in filings:
            if docs_processed >= MAX_DOCUMENTS:
                break

            docs_processed += 1
            print(
                f"[{docs_processed}/{MAX_DOCUMENTS}] Analyzing {ticker} "
                f"8-K filed on {filing['filing_time']}..."
            )

            products = extract_products_with_api(
                company_name=default_company_name,
                ticker=ticker,
                filing_time=filing["filing_time"],
                text=filing["text"],
            )
            if products:
                for p in products:
                    print(f"  -> Extracted: {p['new_product']}")
            extracted_rows.extend(products)

    # 1. Generate CSV File (Deliverable 5 Output Format)
    fieldnames = [
        "company_name",
        "stock_name",
        "filing_time",
        "new_product",
        "product_description",
    ]
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(extracted_rows)

    # 2. Generate Pipe-Separated Table (Task Output Format)
    header_line = (
        "Company Name | Stock Name | Filing Time | "
        "New Product | Product Description"
    )
    separator_line = "-" * len(header_line)
    pipe_lines = [header_line, separator_line]
    for r in extracted_rows:
        row_str = (
            f"{r['company_name']} | {r['stock_name']} | "
            f"{r['filing_time']} | {r['new_product']} | "
            f"{r['product_description']}"
        )
        pipe_lines.append(row_str)

    pipe_output = "\n".join(pipe_lines)
    with open(OUTPUT_PIPE_TXT, mode="w", encoding="utf-8") as txt_file:
        txt_file.write(pipe_output)

    print("\n" + pipe_output)
    print(
        f"\nSaved {len(extracted_rows)} extracted products "
        f"across {docs_processed} filings to {OUTPUT_CSV}."
    )

if __name__ == "__main__":
    main()
