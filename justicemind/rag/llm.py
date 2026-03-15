"""LLM answer generation for JusticeMind using Google Gemini 2.5."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client

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
    """Call Gemini 2.5 with context and question; return the cited answer string."""
    prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    resp = _get_client().models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.1,
            max_output_tokens=600,
        ),
    )
    return resp.text.strip()
