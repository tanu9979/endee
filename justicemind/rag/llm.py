"""LLM answer generation for JusticeMind using Groq llama-3.1-8b-instant."""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_SYSTEM = (
    "You are JusticeMind, a precise AI legal assistant. "
    "Answer the user's question using ONLY the numbered context passages provided below. "
    "Cite your sources using bracket numbers like [1] or [2]. "
    "When referencing specific clauses, section numbers, or legal provisions, quote them exactly as they appear in the context. "
    "If the context does not contain sufficient information to answer the question, respond with: "
    "The uploaded documents do not contain enough information to answer this question. "
    "Never interpret, infer, or invent legal information not explicitly stated in the context. "
    "Always end your answer with: Disclaimer: This response is for informational purposes only and does not constitute legal advice."
)


def generate_answer(question: str, context: str) -> str:
    """Call Groq LLM with context and question; return the cited answer string."""
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=600,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()
