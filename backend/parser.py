import pdfplumber
import pandas as pd
import io

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
        })
    return transactions

def parse_pdf(file_bytes):
    transactions = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                headers = [str(h).strip().lower() for h in table[0]]
                for row in table[1:]:
                    if row and any(row):
                        row_dict = dict(zip(headers, row))
                        transactions.append({
                            "date": str(row_dict.get("date", "")),
                            "description": str(row_dict.get("description", row_dict.get("details", "Unknown"))),
                            "amount": float(str(row_dict.get("amount", row_dict.get("debit", 0))).replace(",", "") or 0),
                            "currency": str(row_dict.get("currency", "USD")),
                        })
    return transactions

def parse_statement(filename, file_bytes):
    if filename.endswith(".csv"):
        return parse_csv(file_bytes)
    elif filename.endswith(".pdf"):
        return parse_pdf(file_bytes)
    else:
        raise ValueError("Unsupported file type. Use CSV or PDF.")
