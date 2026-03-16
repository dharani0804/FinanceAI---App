import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_ai_response(user_message: str, transactions: list) -> str:
    transaction_summary = "\n".join([
        f"- {t['date']}: {t['description']} | {t['currency']} {t['amount']}"
        for t in transactions[:50]
    ])

    system_prompt = f"""You are FinanceAI, a personal finance assistant.
You have access to the user's recent transactions below.
Analyse their spending patterns, answer questions, identify root causes of overspending,
and give practical financial advice. Be concise, friendly and specific.

USER'S TRANSACTIONS:
{transaction_summary if transaction_summary else "No transactions uploaded yet."}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return message.content[0].text
