"""A live-web corpus for the research agent: Wikipedia search plus fetching of any public page.

It implements the same interface as the offline Corpus (search, get, fingerprint), so the agent runtime,
its contracts, grounding checks and injection screening work unchanged on real web content."""

from __future__ import annotations
import os
import time
import httpx
import socket
import ipaddress
import threading
from collections.abc import Callable
from urllib.parse import quote, urlsplit
from ares.tools.corpus import Document, normalize_url

# Wikimedia's robot policy requires a User-Agent with contact information; set HUB_CONTACT_URL to yours.
USER_AGENT = f"ARES-Hub/1.0 ({os.getenv('HUB_CONTACT_URL', 'https://github.com/ares-hub')}; research agent)"
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
MAX_TEXT = 20000


class BlockedURL(ValueError):
    pass


def check_public(url: str, resolver: Callable[[str], list[str]] | None = None) -> None:
    """Refuse non-HTTP(S) schemes and hosts that resolve to private, loopback or link-local addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL(f"Only public http(s) URLs can be fetched: {url}")
    resolve = resolver or (lambda h: [ai[4][0] for ai in socket.getaddrinfo(h, None)])
    try:
        addrs = resolve(parts.hostname)
    except OSError as e:
        raise BlockedURL(f"Cannot resolve {parts.hostname}: {e}") from e
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise BlockedURL(f"{parts.hostname} resolves to a non-public address ({a}); blocked")


def extract_text(html: str, url: str) -> tuple[str, str]:
    import trafilatura
    from lxml import html as lhtml

    title = ""
    try:
        t = lhtml.fromstring(html).xpath("//title/text()")
        title = " ".join(t[0].split()) if t else ""
    except Exception:
        pass
    text = trafilatura.extract(html, url=url, include_tables=True, favor_recall=True) or ""
    return title, text


class LiveWeb:
    """search() uses Wikipedia's search API; get() fetches and extracts any public page (cached per session)."""

    def __init__(self, lang: str = "en", http: httpx.Client | None = None, resolver=None,
                 extra_urls: list[str] | None = None):
        self.lang = lang
        self.http = http or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)
        self.resolver = resolver
        self._cache: dict[str, Document] = {}
        self._lock = threading.Lock()
        self.extra_urls = extra_urls or []
        self.stats = {"searches": 0, "fetches": 0, "cache_hits": 0, "blocked": 0}

    def __len__(self) -> int:
        return len(self._cache)

    def fingerprint(self) -> str:
        return f"live-web-{self.lang}-{time.strftime('%Y%m%d')}"

    def search(self, query: str, k: int = 5) -> list[Document]:
        self.stats["searches"] += 1
        r = self.http.get(WIKI_API.format(lang=self.lang), params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": k, "format": "json",
            "srprop": "snippet"})
        r.raise_for_status()
        out = []
        for hit in r.json().get("query", {}).get("search", []):
            title = hit["title"]
            url = f"https://{self.lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            snippet = _strip_tags(hit.get("snippet", ""))
            out.append(Document(url=url, title=title, text=snippet))
        for u in self.extra_urls:  # user-supplied sources always appear as candidates
            if not any(normalize_url(d.url) == normalize_url(u) for d in out):
                out.append(Document(url=u, title=f"User-provided source: {u}", text="(fetch to read)"))
        return out[: k + len(self.extra_urls)]

    def get(self, url: str) -> Document | None:
        key = normalize_url(url)
        with self._lock:
            if key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[key]
        try:
            check_public(url, self.resolver)
        except BlockedURL:
            self.stats["blocked"] += 1
            raise
        self.stats["fetches"] += 1
        r = self.http.get(url)
        if r.status_code >= 400:
            return None
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and ctype:
            return None
        title, text = extract_text(r.text, str(r.url))
        if not text.strip():
            return None
        doc = Document(url=url, title=title or url, text=text[:MAX_TEXT])
        with self._lock:
            self._cache[key] = doc
        return doc


def _strip_tags(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s).replace("&quot;", '"').replace("&amp;", "&")
