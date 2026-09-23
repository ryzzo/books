import re
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
SUMMARY_WORD_LIMIT = 200

_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline

        _summarizer = pipeline("summarization", model=SUMMARY_MODEL)
    return _summarizer


def _clean_summary_text(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _truncate_to_sentences(text: str, word_limit: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    word_count = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if kept and word_count + sentence_words > word_limit:
            break
        kept.append(sentence)
        word_count += sentence_words
    return " ".join(kept) if kept else text


def summarize_text(text: str, word_limit: int = SUMMARY_WORD_LIMIT) -> str:
    if not text.strip():
        return text

    summarizer = get_summarizer()
    # Give the model room to finish its last sentence naturally; we trim to
    # the word limit afterwards on a sentence boundary rather than mid-sentence.
    max_tokens = int(word_limit * 1.6)
    min_tokens = int(word_limit * 0.6)
    result = summarizer(
        text,
        max_length=max_tokens,
        min_length=min_tokens,
        truncation=True,
        do_sample=False,
    )
    summary = _clean_summary_text(result[0]["summary_text"])
    return _truncate_to_sentences(summary, word_limit)


class BookRequest(BaseModel):
    title: str


class BookSummary(BaseModel):
    title: str
    summary: str
    url: Optional[str] = None
    genre: Optional[str] = None
    rating: Optional[str] = None
    image_url: Optional[str] = None


def _split_list_template(content: str) -> str:
    if "\n" in content:
        items = [line.strip().lstrip("*").strip() for line in content.splitlines()]
    else:
        items = content.split("|")
    return ", ".join(item for item in items if item)


def _clean_wikitext_value(value: str) -> str:
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"<ref[^/]*/>", "", value, flags=re.IGNORECASE)
    # Resolve wikilinks first so a link's own "|" (target|display) isn't mistaken
    # for a list-template item separator below.
    value = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", value)
    value = re.sub(
        r"\{\{\s*(?:Plainlist|Flatlist|Unbulleted list|ubl|hlist)\s*\|(.*?)\}\}",
        lambda m: _split_list_template(m.group(1)),
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"'''?", "", value)
    value = re.sub(r"\s*<br\s*/?>\s*", ", ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+([,;])", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,")


def _extract_infobox_field(wikitext: str, field_names: list[str]) -> Optional[str]:
    for name in field_names:
        pattern = rf"\|\s*{re.escape(name)}\s*=\s*(.*?)(?=\n\s*\||\n\}}\}}|\Z)"
        match = re.search(pattern, wikitext, re.IGNORECASE | re.DOTALL)
        if match:
            value = _clean_wikitext_value(match.group(1))
            if value:
                return value
    return None


BOOK_INFOBOX_PATTERN = re.compile(r"\{\{\s*infobox\s+(?:book|novel)\b", re.IGNORECASE)


def _get_wikitext(page: dict) -> str:
    revisions = page.get("revisions") or [{}]
    return revisions[0].get("slots", {}).get("main", {}).get("content", "")


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
        "prop": "extracts|pageimages|revisions",
        "explaintext": "1",
        "exintro": "1",
        "redirects": "1",
        "piprop": "thumbnail",
        "pithumbsize": "500",
        "rvprop": "content",
        "rvslots": "main",
        "rvsection": "0",
        "formatversion": "2",
        "titles": title,
    }
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        response = await client.get(WIKIPEDIA_API_URL, params=params)
    response.raise_for_status()

    pages = response.json().get("query", {}).get("pages", [])
    page = pages[0] if pages else None
    if page is None or page.get("missing"):
        suggestions = await fetch_wikipedia_suggestions(title)
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No Wikipedia page found for '{title}'",
                "suggestions": suggestions,
            },
        )

    wikitext = _get_wikitext(page)
    if not BOOK_INFOBOX_PATTERN.search(wikitext):
        suggestions = await fetch_wikipedia_suggestions(title)
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"'{title}' doesn't appear to be a book on Wikipedia",
                "suggestions": suggestions,
            },
        )

    page_title = page.get("title", title)
    extract = page.get("extract", "")
    image_url = page.get("thumbnail", {}).get("source")
    genre = _extract_infobox_field(wikitext, ["genre"])
    rating = _extract_infobox_field(wikitext, ["rating", "stars", "score", "metascore"])
    summary = await run_in_threadpool(summarize_text, extract)
    return BookSummary(
        title=page_title,
        summary=summary,
        url=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
        genre=genre,
        rating=rating,
        image_url=image_url,
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
