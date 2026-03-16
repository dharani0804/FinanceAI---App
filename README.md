# FinanceAI — Personal Finance Tracker

A full-stack AI-powered personal finance tracker built with React, FastAPI, and Claude AI.

## Features

- 📊 **Dashboard** — Visual overview with scorecards, bar chart and pie chart
- 📎 **Statement Upload** — Import bank statements via CSV or PDF
- 💬 **AI Chatbot** — Ask Claude anything about your spending
- 🎯 **Goals** — Set and track financial goals
- 💱 **Multi-currency** — Supports USD, EUR, GBP, INR, AUD and more

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Backend | Python FastAPI |
| Database | SQLite + SQLAlchemy |
| AI | Claude API (claude-sonnet-4-6) |
| Charts | Recharts |
| Icons | Lucide React |

## Project Structure
```
financeai/
├── frontend/          # React PWA
│   └── src/
│       ├── App.jsx
│       ├── Navbar.jsx
│       └── pages/
│           ├── Dashboard.jsx
│           ├── Upload.jsx
│           ├── Chat.jsx
│           └── Goals.jsx
└── backend/           # FastAPI server
    ├── main.py        # API endpoints
    ├── models.py      # Database models
    ├── database.py    # SQLite connection
    ├── parser.py      # CSV/PDF parser
    ├── ai.py          # Claude API integration
    └── requirements.txt
```

## Getting Started

### Prerequisites
- Node.js v18+
- Python 3.12+
- Anthropic API key

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file
echo 'ANTHROPIC_API_KEY=your-key-here' > .env

uvicorn main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Usage
1. Open http://localhost:5173
2. Go to **Upload** and import a bank statement (CSV or PDF)
3. View your spending breakdown on the **Dashboard**
4. Chat with **AI Chat** to analyse your finances
5. Set savings targets in **Goals**

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload bank statement |
| GET | `/transactions` | Get all transactions |
| PATCH | `/transactions/{id}` | Edit a transaction |
| GET | `/banks` | List all bank sources |
| GET | `/summary` | Financial summary |
| POST | `/chat` | AI chat |
| GET | `/categories` | List categories |

## Roadmap

- [ ] Multi-statement upload with bank tagging
- [ ] Transaction editing and categorisation
- [ ] AI-powered goal recommendations
- [ ] External financial data integration
- [ ] PWA offline support
- [ ] Deploy to Vercel + Render

## Built With

- [Anthropic Claude API](https://anthropic.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [React](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [Recharts](https://recharts.org)

---

Built step by step with Claude AI assistance.
