from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from models import Transaction
from parser import parse_statement
from ai import get_ai_response
from datetime import datetime
from pydantic import BaseModel

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinanceAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "FinanceAI backend is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_statement(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
                amount=t["amount"],
                currency=t["currency"],
                category="Uncategorized"
            )
            db.add(tx)
            saved.append(t)
        db.commit()
        return {"success": True, "transactions_imported": len(saved), "transactions": saved}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/transactions")
def get_transactions(db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    return [
        {
            "id": t.id,
            "date": str(t.date),
            "description": t.description,
            "amount": t.amount,
            "currency": t.currency,
            "category": t.category
        }
        for t in transactions
    ]

@app.post("/chat")
def chat(msg: ChatMessage, db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    tx_list = [
        {
            "date": str(t.date),
            "description": t.description,
            "amount": t.amount,
            "currency": t.currency,
            "category": t.category
        }
        for t in transactions
    ]
    try:
        response = get_ai_response(msg.message, tx_list)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}
