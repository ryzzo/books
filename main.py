from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Book Summary API")

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "books-api/1.0 (https://github.com/example/books-api)"}


class BookRequest(BaseModel):
    title: str


class BookSummary(BaseModel):
    title: str
    summary: str
    url: Optional[str] = None


async def fetch_wikipedia_summary(title: str) -> BookSummary:
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        response = await client.get(WIKIPEDIA_SUMMARY_URL.format(title=title))

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"No Wikipedia page found for '{title}'")
    response.raise_for_status()

    data = response.json()
    return BookSummary(
        title=data.get("title", title),
        summary=data.get("extract", ""),
        url=data.get("content_urls", {}).get("desktop", {}).get("page"),
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


@app.get("/")
async def root():
    return {"docs": "/docs", "example": "/books/Dune/summary"}
