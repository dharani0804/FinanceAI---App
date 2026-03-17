# FinanceAI 💰

> Know exactly where your money goes — without spreadsheets, without linking your bank, without the complexity.

FinanceAI is a personal finance tracker powered by Claude AI. Upload a bank statement or enter transactions manually, and instantly get a clear picture of your spending — with an AI assistant you can actually talk to.

---

## The Problem We're Solving

Most people don't know where their money actually goes.

- Bank apps show raw transactions but offer no insight
- Budget spreadsheets need manual input every time — and most people fall behind
- Switching between multiple bank accounts makes it even harder to see the full picture
- Understanding *why* you overspent is still a completely manual exercise

FinanceAI fixes all of this — automatically.

---

## What It Does

### 📎 Upload Any Bank Statement
Drop in a CSV or PDF from any bank, anywhere in the world. FinanceAI parses it instantly — no manual entry, no bank credentials, no linking required. Your statement is never stored; only the transaction data is saved.

### ✏️ Or Enter Transactions Manually
Prefer to add transactions one by one? Use the manual entry form. Every field — date, merchant, amount, currency, category — is editable after saving too.

### 📊 Instant Visual Dashboard
The moment you upload, your dashboard comes alive:
- **Scorecards** — total balance, income, spending, savings rate
- **Monthly bar chart** — income vs spending side by side
- **Category pie chart** — where your money actually goes
- **Transaction table** — every entry, fully editable inline

### 💬 AI Chat Assistant
Ask Claude AI anything about your finances in plain English:
- *"Why did I overspend in January?"*
- *"Where is most of my money going?"*
- *"How much can I realistically save each month?"*

Claude analyses your actual transactions and gives specific, data-driven answers — not generic tips.

### 🎯 Goals & Recommendations
Set savings targets with deadlines. Get AI-generated recommendations tailored to your real spending patterns.

### 💱 Multi-Currency
Works with USD, EUR, GBP, INR, AUD, SGD, AED and 150+ more. Built for people with accounts across multiple countries.

### 🔒 Private by Default
Your data never leaves your device. No cloud database, no third-party sharing — everything stored locally.

---

## How It Works
```
1. Upload your bank statement (CSV or PDF)
        ↓
2. FinanceAI parses and categorises every transaction
        ↓
3. Your dashboard populates instantly with charts and metrics
        ↓
4. Chat with your AI assistant to understand your spending
        ↓
5. Set goals and get personalised recommendations
```

---

## Current Status

| Feature | Status |
|---|---|
| Landing page | ✅ Live |
| Statement upload (CSV + PDF) | ✅ Live |
| Manual transaction entry | ✅ Live |
| Inline transaction editing | ✅ Live |
| Dashboard with charts | ✅ Live |
| AI chat assistant | ✅ Live |
| Multi-currency support | ✅ Live |
| Goals page | 🔲 In progress |
| Multi-bank tracking | 🔲 In progress |
| AI goal recommendations | 🔲 In progress |
| User auth (login / signup) | 🔲 Planned |
| Deploy to web | 🔲 Planned |

---

## Roadmap

**Next up**
- [ ] Goals page with progress tracking
- [ ] AI-powered savings recommendations
- [ ] Multi-statement tracking by bank
- [ ] Category-level spending limits and alerts

**Coming later**
- [ ] User authentication (login / signup)
- [ ] Optional cloud sync and backup
- [ ] Mobile-friendly PWA
- [ ] Deploy to web — shareable link

---

## Pricing

| Plan | Price | What's included |
|---|---|---|
| Free | $0 forever | 1 upload/mo, 50 transactions, 10 AI messages, 1 goal |
| Plus | $4.99/mo | Unlimited uploads, full dashboard, unlimited AI chat, 5 goals |
| Pro | $9.99/mo | Everything in Plus + multi-bank, AI reports, priority support |

---

## Built With

- **Frontend** — React 18, Vite, Tailwind CSS
- **Backend** — Python, FastAPI, SQLite
- **AI** — Claude API by Anthropic (claude-sonnet-4-6)
- **Charts** — Recharts
- **Icons** — Lucide React

---

## Getting Started (Local)
```bash
# Clone the repo
git clone https://github.com/dharani0804/FinanceAI---App.git
cd FinanceAI---App

# Start the backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
uvicorn main:app --reload

# Start the frontend (new terminal)
cd frontend
npm install
npm run dev

# Open in browser
http://localhost:5173
```

Get a free Claude API key at **https://console.anthropic.com**

---

## Why Not Just Use a Spreadsheet?

| | Spreadsheet | FinanceAI |
|---|---|---|
| Manual data entry | Every time | Never |
| Understands your spending | You figure it out | AI explains it |
| Works across multiple banks | Copy-paste nightmare | Upload and done |
| Ask questions about your money | Not possible | Just type and ask |
| Setup time | Hours | 30 seconds |

---

*Built step by step with Claude AI · © 2026 FinanceAI*
