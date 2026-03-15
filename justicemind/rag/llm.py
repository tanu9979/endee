"""LLM answer generation for JusticeMind using Google Gemini 2.5."""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel(
    model_name="gemini-2.5-pro-exp-03-25",
    system_instruction=(
        "You are JusticeMind, a precise AI legal assistant. "
        "Answer the user's question using ONLY the numbered context passages provided below. "
        "Cite your sources using bracket numbers like [1] or [2]. "
        "When referencing specific clauses, section numbers, or legal provisions, quote them exactly as they appear in the context. "
        "If the context does not contain sufficient information to answer the question, respond with: "
        "The uploaded documents do not contain enough information to answer this question. "
        "Never interpret, infer, or invent legal information not explicitly stated in the context. "
        "Always end your answer with: Disclaimer: This response is for informational purposes only and does not constitute legal advice."
    ),
)


def generate_answer(question: str, context: str) -> str:
    """Call Gemini 2.5 with context and question; return the cited answer string."""
    prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    resp = _model.generate_content(prompt)
    return resp.text.strip()
