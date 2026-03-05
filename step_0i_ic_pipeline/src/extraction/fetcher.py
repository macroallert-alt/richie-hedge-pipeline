"""
IC Pipeline — Stufe 1: Content Fetcher
YouTube Transcripts, RSS Feeds, Web Scraper
"""

import json
import logging
import os
import re
import time
from datetime import datetime, date, timedelta
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FETCH_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "history", "fetch_state.json"
)
MAX_CONTENT_LENGTH = 80_000  # chars — safety cap for LLM input


# ---------------------------------------------------------------------------
# Fetch State Persistence
# ---------------------------------------------------------------------------
def load_fetch_state() -> dict:
    path = os.path.normpath(FETCH_STATE_PATH)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"fetch_state": {}, "last_updated": None}


def save_fetch_state(state: dict) -> None:
    path = os.path.normpath(FETCH_STATE_PATH)
    state["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# YouTube Transcript Fetcher
# ---------------------------------------------------------------------------
def _get_channel_id_from_url(channel_url: str) -> Optional[str]:
    """Extract channel ID from YouTube channel URL via page scrape."""
    try:
        resp = requests.get(channel_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; IC-Pipeline/1.0)"
        })
        resp.raise_for_status()
        match = re.search(r'"externalId":"(UC[^"]+)"', resp.text)
        if match:
            return match.group(1)
        match = re.search(r'channel_id=([^"&]+)', resp.text)
        if match:
            return match.group(1)
    except Exception as e:
        logger.warning(f"Could not extract channel ID from {channel_url}: {e}")
    return None


def _get_latest_video_ids(channel_url: str, max_videos: int = 3) -> list[dict]:
    """Get latest video IDs from YouTube channel RSS feed."""
    channel_id = _get_channel_id_from_url(channel_url)
    if not channel_id:
        logger.warning(f"No channel_id for {channel_url}")
        return []

    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    feed = feedparser.parse(rss_url)

    results = []
    for entry in feed.entries[:max_videos]:
        video_id = entry.get("yt_videoid", "")
        if not video_id:
            link = entry.get("link", "")
            match = re.search(r"v=([^&]+)", link)
            video_id = match.group(1) if match else ""

        if video_id:
            pub_date = entry.get("published", "")
            try:
                dt = datetime.strptime(pub_date[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                dt = date.today()

            results.append({
                "video_id": video_id,
                "title": entry.get("title", ""),
                "published": dt.isoformat(),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            })
    return results


def fetch_youtube_transcript(source: dict, fetch_state: dict) -> Optional[dict]:
    """
    Fetch latest YouTube transcript for a source.
    Returns dict with source_id, content_date, title, text, url or None.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    source_id = source["source_id"]
    channel_url = source["fetch_config"]["channel_url"]
    lang = source["fetch_config"].get("language", "en")

    last_content_id = fetch_state.get("fetch_state", {}).get(
        source_id, {}
    ).get("last_content_id", "")

    videos = _get_latest_video_ids(channel_url, max_videos=3)
    if not videos:
        logger.info(f"[{source_id}] No videos found")
        return None

    for video in videos:
        vid = video["video_id"]
        if vid == last_content_id:
            logger.info(f"[{source_id}] No new video (last={last_content_id})")
            return None

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                vid, languages=[lang, "en"]
            )
            text = " ".join(seg["text"] for seg in transcript_list)
            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH]

            # Update fetch state
            fetch_state.setdefault("fetch_state", {})[source_id] = {
                "last_fetch_date": date.today().isoformat(),
                "last_content_id": vid,
                "last_content_date": video["published"],
            }

            return {
                "source_id": source_id,
                "source_name": source["source_name"],
                "content_date": video["published"],
                "title": video["title"],
                "text": text,
                "url": video["url"],
                "content_type": "podcast",
                "fetch_method": "youtube_transcript",
            }
        except Exception as e:
            logger.warning(f"[{source_id}] Transcript failed for {vid}: {e}")
            continue

    return None


# ---------------------------------------------------------------------------
# RSS Feed Fetcher
# ---------------------------------------------------------------------------
def _html_to_text(html: str) -> str:
    """Strip HTML tags, return clean text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _scrape_full_article(url: str) -> Optional[str]:
    """Scrape full article text from URL."""
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; IC-Pipeline/1.0)"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try common article selectors
        for selector in ["article", ".post-content", ".entry-content",
                         ".article-body", "main", ".content"]:
            el = soup.select_one(selector)
            if el and len(el.get_text(strip=True)) > 200:
                return el.get_text(separator=" ", strip=True)

        # Fallback: largest text block
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        if len(text) > 200:
            return text

    except Exception as e:
        logger.warning(f"Scrape failed for {url}: {e}")
    return None


def fetch_rss(source: dict, fetch_state: dict) -> list[dict]:
    """
    Fetch new RSS entries for a source.
    Returns list of content dicts (may be multiple articles).
    """
    source_id = source["source_id"]
    config = source["fetch_config"]
    rss_url = config.get("rss_url", "")

    if not rss_url or rss_url == "TO_BE_CONFIGURED":
        logger.info(f"[{source_id}] RSS not configured")
        return []

    full_text_in_rss = config.get("full_text_in_rss", False)
    scrape_full = config.get("scrape_full_article", False)
    max_articles = config.get("max_articles_per_day", 10)

    last_content_date = fetch_state.get("fetch_state", {}).get(
        source_id, {}
    ).get("last_content_date", "2000-01-01")

    try:
        last_dt = datetime.strptime(last_content_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        last_dt = date(2000, 1, 1)

    feed = feedparser.parse(rss_url)
    if feed.bozo and not feed.entries:
        logger.warning(f"[{source_id}] RSS parse error: {feed.bozo_exception}")
        return []

    results = []
    latest_date = last_dt

    for entry in feed.entries[:max_articles]:
        # Parse date
        pub = entry.get("published_parsed") or entry.get("updated_parsed")
        if pub:
            entry_date = date(pub.tm_year, pub.tm_mon, pub.tm_mday)
        else:
            entry_date = date.today()

        if entry_date <= last_dt:
            continue

        # Get text
        text = ""
        if full_text_in_rss:
            content = entry.get("content", [{}])
            if content:
                raw = content[0].get("value", "")
                text = _html_to_text(raw)
            if not text:
                text = _html_to_text(entry.get("summary", ""))
        else:
            text = _html_to_text(entry.get("summary", ""))

        # If not enough text and scraping enabled
        link = entry.get("link", "")
        if (not text or len(text) < 300) and scrape_full and link:
            scraped = _scrape_full_article(link)
            if scraped:
                text = scraped

        if not text or len(text) < 100:
            logger.info(f"[{source_id}] Skipping entry with insufficient text: {entry.get('title', 'untitled')}")
            continue

        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH]

        if entry_date > latest_date:
            latest_date = entry_date

        content_type = "newsletter" if "substack" in rss_url else "blog"

        results.append({
            "source_id": source_id,
            "source_name": source["source_name"],
            "content_date": entry_date.isoformat(),
            "title": entry.get("title", "Untitled"),
            "text": text,
            "url": link,
            "content_type": content_type,
            "fetch_method": "rss",
        })

    # Update fetch state
    if results:
        last_url = results[0]["url"]
        fetch_state.setdefault("fetch_state", {})[source_id] = {
            "last_fetch_date": date.today().isoformat(),
            "last_content_id": last_url,
            "last_content_date": latest_date.isoformat(),
        }

    return results


# ---------------------------------------------------------------------------
# Web Scraper Fetcher (Fallback)
# ---------------------------------------------------------------------------
def fetch_web_scrape(source: dict, fetch_state: dict) -> list[dict]:
    """
    Fallback web scraper for sources without RSS/YouTube.
    Currently delegates to RSS with scraping enabled.
    """
    return fetch_rss(source, fetch_state)


# ---------------------------------------------------------------------------
# Main Fetcher Dispatcher
# ---------------------------------------------------------------------------
def fetch_source(source: dict, fetch_state: dict) -> list[dict]:
    """
    Fetch content for a single source based on its fetch_method.
    Returns list of content dicts.
    """
    method = source.get("fetch_method", "")
    source_id = source["source_id"]

    if not source.get("active", True):
        logger.info(f"[{source_id}] Source inactive, skipping")
        return []

    logger.info(f"[{source_id}] Fetching via {method}...")

    try:
        if method == "youtube":
            result = fetch_youtube_transcript(source, fetch_state)
            return [result] if result else []
        elif method == "rss":
            return fetch_rss(source, fetch_state)
        elif method == "web_scrape":
            return fetch_web_scrape(source, fetch_state)
        else:
            logger.warning(f"[{source_id}] Unknown fetch method: {method}")
            return []
    except Exception as e:
        logger.error(f"[{source_id}] Fetch failed: {e}")
        return []


def fetch_all_sources(sources: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """
    Fetch content from all active sources.
    Returns: (all_content, fetch_state, failed_sources)
    """
    fetch_state = load_fetch_state()
    all_content = []
    failed_sources = []

    for source in sources:
        if not source.get("active", True):
            continue

        try:
            content_list = fetch_source(source, fetch_state)
            if content_list:
                all_content.extend(content_list)
                logger.info(
                    f"[{source['source_id']}] Fetched {len(content_list)} item(s)"
                )
            else:
                logger.info(f"[{source['source_id']}] No new content")
        except Exception as e:
            logger.error(f"[{source['source_id']}] FAILED: {e}")
            failed_sources.append({
                "source_id": source["source_id"],
                "error": str(e),
                "retry_next_run": True,
            })

    save_fetch_state(fetch_state)
    return all_content, fetch_state, failed_sources