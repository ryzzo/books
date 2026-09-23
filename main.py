from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Book Summary API")

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "books-api/1.0 (https://github.com/example/books-api)"}

SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
SUMMARY_WORD_LIMIT = 100

_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline

        _summarizer = pipeline("summarization", model=SUMMARY_MODEL)
    return _summarizer


def summarize_text(text: str, word_limit: int = SUMMARY_WORD_LIMIT) -> str:
    if not text.strip():
        return text

    summarizer = get_summarizer()
    # ~1.3 tokens per word for BART's tokenizer; cap generation accordingly.
    max_tokens = int(word_limit * 1.3)
    min_tokens = int(word_limit * 0.8)
    result = summarizer(
        text,
        max_length=max_tokens,
        min_length=min_tokens,
        truncation=True,
        do_sample=False,
    )
    summary = result[0]["summary_text"].strip()

    words = summary.split()
    if len(words) > word_limit:
        summary = " ".join(words[:word_limit]).rstrip(",;:") + "."
    return summary


class BookRequest(BaseModel):
    title: str


class BookSummary(BaseModel):
    title: str
    summary: str
    url: Optional[str] = None


async def fetch_wikipedia_suggestions(title: str) -> list[str]:
    params = {
        "action": "opensearch",
        "format": "json",
        "search": title,
        "limit": "5",
        "namespace": "0",
    }
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        response = await client.get(WIKIPEDIA_API_URL, params=params)
    response.raise_for_status()

    result = response.json()
    return result[1] if len(result) > 1 else []


async def fetch_wikipedia_summary(title: str) -> BookSummary:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exintro": "1",
        "redirects": "1",
        "titles": title,
    }
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        response = await client.get(WIKIPEDIA_API_URL, params=params)
    response.raise_for_status()

    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if page is None or "missing" in page:
        suggestions = await fetch_wikipedia_suggestions(title)
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No Wikipedia page found for '{title}'",
                "suggestions": suggestions,
            },
        )

    page_title = page.get("title", title)
    extract = page.get("extract", "")
    summary = await run_in_threadpool(summarize_text, extract)
    return BookSummary(
        title=page_title,
        summary=summary,
        url=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
    )


@app.get("/books/{title}/summary", response_model=BookSummary)
async def get_book_summary_by_path(title: str):
    return await fetch_wikipedia_summary(title)


@app.post("/books/summary", response_model=BookSummary)
async def get_book_summary(book: BookRequest):
    return await fetch_wikipedia_summary(book.title)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
