from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from models import Transaction
from parser import parse_statement
from ai import get_ai_response
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinanceAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = [
    "Uncategorized","Food & Dining","Shopping","Transport",
    "Entertainment","Bills & Utilities","Health","Travel",
    "Groceries","Salary","Transfer","Other"
]

class ChatMessage(BaseModel):
    message: str

class ManualTransaction(BaseModel):
    date: str
    description: str
    amount: float
    currency: str = "USD"
    category: str = "Uncategorized"
    bank_source: str = "Manual Entry"

class TransactionUpdate(BaseModel):
    date: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    bank_source: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "FinanceAI backend is running"}

@app.get("/categories")
def get_categories():
    return CATEGORIES

@app.post("/upload")
async def upload_statement(
    file: UploadFile = File(...),
    bank_name: str = "Unknown Bank",
    db: Session = Depends(get_db)
):
    contents = await file.read()
    try:
        transactions = parse_statement(file.filename, contents)
        saved = []
        for t in transactions:
            try:
                date = datetime.strptime(t["date"], "%Y-%m-%d").date() if t["date"] else None
            except:
                date = None
            tx = Transaction(
                date=date,
                description=t["description"],
                original_description=t["description"],
                amount=t["amount"],
                currency=t["currency"],
                category="Uncategorized",
                bank_source=bank_name,
                is_edited=False
            )
            db.add(tx)
            saved.append(t)
        db.commit()
        return {"success": True, "transactions_imported": len(saved), "transactions": saved}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/transactions/manual")
def add_manual_transaction(tx: ManualTransaction, db: Session = Depends(get_db)):
    try:
        try:
            date = datetime.strptime(tx.date, "%Y-%m-%d").date()
        except:
            date = datetime.today().date()
        new_tx = Transaction(
            date=date,
            description=tx.description,
            original_description=tx.description,
            amount=tx.amount,
            currency=tx.currency,
            category=tx.category,
            bank_source=tx.bank_source,
            is_edited=False
        )
        db.add(new_tx)
        db.commit()
        db.refresh(new_tx)
        return {
            "success": True,
            "transaction": {
                "id": new_tx.id,
                "date": str(new_tx.date),
                "description": new_tx.description,
                "amount": new_tx.amount,
                "currency": new_tx.currency,
                "category": new_tx.category,
                "bank_source": new_tx.bank_source,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/transactions")
def get_transactions(db: Session = Depends(get_db)):
    transactions = db.query(Transaction).order_by(Transaction.date.desc()).all()
    return [
        {
            "id": t.id,
            "date": str(t.date),
            "description": t.description,
            "original_description": t.original_description,
            "amount": t.amount,
            "currency": t.currency,
            "category": t.category,
            "bank_source": t.bank_source,
            "is_edited": t.is_edited
        }
        for t in transactions
    ]

@app.patch("/transactions/{transaction_id}")
def update_transaction(
    transaction_id: int,
    update: TransactionUpdate,
    db: Session = Depends(get_db)
):
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if update.date is not None:
        try:
            tx.date = datetime.strptime(update.date, "%Y-%m-%d").date()
        except:
            pass
    if update.description is not None:
        tx.description = update.description
    if update.amount is not None:
        tx.amount = update.amount
    if update.currency is not None:
        tx.currency = update.currency
    if update.category is not None:
        tx.category = update.category
    if update.bank_source is not None:
        tx.bank_source = update.bank_source
    tx.is_edited = True
    db.commit()
    return {
        "id": tx.id,
        "date": str(tx.date),
        "description": tx.description,
        "amount": tx.amount,
        "currency": tx.currency,
        "category": tx.category,
        "bank_source": tx.bank_source,
        "is_edited": tx.is_edited
    }

@app.delete("/transactions/{transaction_id}")
def delete_transaction(transaction_id: int, db: Session = Depends(get_db)):
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    db.delete(tx)
    db.commit()
    return {"success": True}

@app.post("/chat")
def chat(msg: ChatMessage, db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    tx_list = [
        {
            "date": str(t.date),
            "description": t.description,
            "amount": t.amount,
            "currency": t.currency,
            "category": t.category,
            "bank_source": t.bank_source
        }
        for t in transactions
    ]
    try:
        response = get_ai_response(msg.message, tx_list)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}
