import anthropic
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CATEGORIES = [
    "Food & Dining", "Shopping", "Transport", "Entertainment",
    "Bills & Utilities", "Health", "Travel", "Groceries",
    "Salary", "Transfer", "Payment", "Other"
]

def extract_transactions_from_chunk(chunk: str) -> list:
    prompt = f"""Extract ALL transactions from this bank statement text. Return ONLY a JSON array, nothing else.

Each item MUST have these exact fields:
{{"date":"YYYY-MM-DD","description":"merchant name","amount":-45.00,"currency":"USD","category":"Food & Dining"}}

Category must be one of: Food & Dining, Shopping, Transport, Entertainment, Bills & Utilities, Health, Travel, Groceries, Salary, Transfer, Payment, Other

Rules:
- Negative amount = expense/purchase
- Positive amount = payment/credit/income  
- Skip anything that is not a transaction (headers, summaries, APR tables, disclaimers)
- Every transaction MUST include a category

Return ONLY the JSON array. No explanation. No markdown.

Text:
{chunk}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    # Handle empty array
    if not raw or raw == '[]':
        return []

    try:
        transactions = json.loads(raw)
        valid = []
        for t in transactions:
            if all(k in t for k in ['date', 'description', 'amount']):
                valid.append({
                    "date": str(t.get("date", "")),
                    "description": str(t.get("description", "Unknown")),
                    "amount": float(t.get("amount", 0)),
                    "currency": str(t.get("currency", "USD")),
                    "category": str(t.get("category", "Other")),
                })
        return valid
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw: {raw[:200]}")
        return []

def parse_statement_with_claude(raw_text: str, filename: str) -> list:
    """Split large PDFs into chunks and parse each one"""
    
    # Split text into chunks of ~4000 chars with overlap
    chunk_size = 4000
    overlap = 200
    chunks = []
    
    start = 0
    while start < len(raw_text):
        end = start + chunk_size
        chunk = raw_text[start:end]
        chunks.append(chunk)
        start = end - overlap

    print(f"Processing {len(chunks)} chunks...")
    
    all_transactions = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        txs = extract_transactions_from_chunk(chunk)
        print(f"  Found {len(txs)} transactions")
        all_transactions.extend(txs)

    # Deduplicate by date + description + amount
    seen = set()
    unique = []
    for t in all_transactions:
        key = (t['date'], t['description'], round(t['amount'], 2))
        if key not in seen:
            seen.add(key)
            unique.append(t)

    print(f"Total unique transactions: {len(unique)}")
    return unique

def get_ai_response(user_message: str, transactions: list) -> str:
    transaction_summary = "\n".join([
        f"- {t['date']}: {t['description']} | {t['currency']} {t['amount']} | {t.get('category','')}"
        for t in transactions[:50]
    ])

    system_prompt = f"""You are FinanceAI, a personal finance assistant.
You have access to the user's recent transactions below.
Analyse their spending patterns, answer questions, identify root causes of overspending,
and give practical financial advice. Be concise, friendly and specific.
When listing items use plain text, not markdown symbols.

USER'S TRANSACTIONS:
{transaction_summary if transaction_summary else "No transactions uploaded yet."}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    return message.content[0].text
