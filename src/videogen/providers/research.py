from __future__ import annotations

from dataclasses import dataclass

import httpx
import trafilatura
from ddgs import DDGS

from videogen.config import Settings
from videogen.models import Source


@dataclass
class PageExtract:
    url: str
    title: str
    text: str


class WebResearcher:
    """DuckDuckGo search + local page extraction. No model API."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def search(self, query: str) -> list[Source]:
        results: list[Source] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=self.settings.research.max_results):
                results.append(
                    Source(
                        title=item.get("title") or "",
                        url=item.get("href") or item.get("link") or "",
                        snippet=item.get("body") or item.get("snippet") or "",
                    )
                )
        return [s for s in results if s.url]

    def fetch_page(self, url: str) -> PageExtract | None:
        try:
            with httpx.Client(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "videogen-local-research/0.1"},
            ) as client:
                r = client.get(url)
                r.raise_for_status()
                html = r.text
        except httpx.HTTPError:
            return None
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            url=url,
        )
        if not extracted:
            return None
        title = trafilatura.extract_metadata(html)
        title_str = (title.title if title and title.title else url)[:200]
        limit = self.settings.research.max_chars_per_page
        return PageExtract(url=url, title=title_str, text=extracted[:limit])

    def research(self, topic: str) -> tuple[list[Source], str]:
        sources = self.search(topic)
        pages: list[PageExtract] = []
        for src in sources:
            if len(pages) >= self.settings.research.max_pages:
                break
            page = self.fetch_page(src.url)
            if page and page.text.strip():
                pages.append(page)
                if not src.title:
                    src.title = page.title

        lines = [f"# Research notes: {topic}", ""]
        if not pages:
            lines.append("No pages extracted. Use search snippets only.")
            for s in sources:
                lines.append(f"- {s.title}: {s.snippet} ({s.url})")
        else:
            for p in pages:
                lines.append(f"## {p.title}")
                lines.append(f"Source: {p.url}")
                lines.append("")
                lines.append(p.text)
                lines.append("")
        return sources, "\n".join(lines)
