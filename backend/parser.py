import pdfplumber
import pandas as pd
import io
from ai import parse_statement_with_claude

def parse_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [col.strip().lower() for col in df.columns]
    transactions = []
    for _, row in df.iterrows():
        transactions.append({
            "date": str(row.get("date", "")),
            "description": str(row.get("description", row.get("details", row.get("narration", "Unknown")))),
            "amount": float(str(row.get("amount", row.get("debit", 0))).replace(",", "") or 0),
            "currency": str(row.get("currency", "USD")),
            "category": str(row.get("category", "Uncategorized")),
        })
    return transactions

def parse_pdf(file_bytes, filename="statement.pdf"):
    all_text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += "\n" + text
    print("Sending to Claude for parsing...")
    transactions = parse_statement_with_claude(all_text, filename)
    print(f"Claude returned {len(transactions)} transactions")
    return transactions

def parse_statement(filename, file_bytes):
    fname = filename.lower()
    if fname.endswith(".csv"):
        return parse_csv(file_bytes)
    elif fname.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    else:
        raise ValueError("Unsupported file type. Please use CSV or PDF.")
