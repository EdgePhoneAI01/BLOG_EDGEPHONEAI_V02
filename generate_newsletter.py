"""
generate_newsletter.py
======================
EdgePhone.ai – Automated Daily Blog Generator

Pipeline
--------
1. Fetch trending Edge AI headlines from Google News RSS (no API key needed).
2. Build a rich prompt from those headlines.
3. Call the AI API (OpenAI or Gemini) to draft 4 SEO-optimised articles.
4. Open index.html, move the existing "Today's Blogs" articles into the
   "Archived Blogs" section, and inject the 4 new articles.
5. Write the updated index.html back to disk.

Usage
-----
  python generate_newsletter.py

Required environment variables (set as GitHub Actions secrets):
  OPENAI_API_KEY   – your OpenAI key  (leave blank to use Gemini instead)
  GEMINI_API_KEY   – your Google Gemini key

Optional:
  AI_PROVIDER      – "openai" | "gemini"  (default: auto-detect from keys)
  AI_MODEL         – e.g. "gpt-4o" or "gemini-1.5-pro"  (defaults below)
"""

# ── Standard library ──────────────────────────────────────────────
import os
import re
import sys
import json
import html
import logging
import datetime
import textwrap
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

# ── Third-party (installed via requirements.txt) ──────────────────
try:
    import requests
except ImportError:
    requests = None  # fallback to urllib

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit here if needed
# ─────────────────────────────────────────────────────────────────

SITE_URL = "https://www.edgephone.ai/"   # canonical target URL for all CTAs

# Google News RSS search terms used to collect context headlines.
# Each query is fetched separately; the top 3 results per query are kept.
NEWS_TOPICS = [
    "Edge AI",
    "On-device AI",
    "Phone-first AI",
    "Offline AI",
    "Local AI processing",
    "On-device inference",
    "Lightweight AI models",
    "Low-latency AI",
    "Privacy-first AI",
    "Zero-latency AI",
    "Mobile machine learning",
    "Edge ML",
    "Battery-efficient AI",
    "Secure mobile AI",
    "Offline voice processing",
    "On-device computer vision",
    "Sensor fusion mobile",
    "Mobile AI APIs",
    "Edge computing IoT",
    "Local inference AI",
    "Data minimization AI",
    "Privacy by default AI",
]

# How many headlines to sample per topic for the prompt
HEADLINES_PER_TOPIC = 2

# Number of articles to generate per day
ARTICLES_TO_GENERATE = 4

# AI provider config
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"

# Path to the HTML file (relative to script location)
HTML_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
ARTICLES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "articles.json")

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# STEP 1 – FETCH NEWS HEADLINES
# ─────────────────────────────────────────────────────────────────

def _http_get(url: str, timeout: int = 10) -> bytes:
    """Simple GET with a browser-like User-Agent to avoid 403s."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (EdgePhone.ai Blog Bot)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_google_news_rss(topic: str, max_results: int = 3) -> list[dict]:
    """
    Fetch the Google News RSS feed for a search query.
    Returns a list of dicts: {"title": str, "url": str, "published": str}
    """
    encoded = quote_plus(topic)
    rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    headlines = []
    try:
        raw = _http_get(rss_url, timeout=12)
        root = ET.fromstring(raw)
        channel = root.find("channel")
        if channel is None:
            return []
        for item in channel.findall("item")[:max_results]:
            title_el = item.find("title")
            link_el  = item.find("link")
            pub_el   = item.find("pubDate")
            if title_el is not None and link_el is not None:
                headlines.append({
                    "title":     html.unescape(title_el.text or ""),
                    "url":       link_el.text or "",
                    "published": pub_el.text if pub_el is not None else "",
                })
    except (URLError, ET.ParseError, Exception) as exc:
        log.warning("News fetch failed for topic '%s': %s", topic, exc)
    return headlines


def gather_headlines(topics: list[str], per_topic: int = HEADLINES_PER_TOPIC) -> list[dict]:
    """
    Iterate over all topics, collect headlines, and deduplicate by title.
    """
    seen_titles: set[str] = set()
    all_headlines: list[dict] = []

    for topic in topics:
        log.info("Fetching news for: %s", topic)
        results = fetch_google_news_rss(topic, max_results=per_topic)
        for h in results:
            norm = h["title"].lower().strip()
            if norm not in seen_titles:
                seen_titles.add(norm)
                h["topic"] = topic
                all_headlines.append(h)

    log.info("Collected %d unique headlines.", len(all_headlines))
    return all_headlines


# ─────────────────────────────────────────────────────────────────
# STEP 2 – BUILD AI PROMPT
# ─────────────────────────────────────────────────────────────────

def build_prompt(headlines: list[dict], num_articles: int = ARTICLES_TO_GENERATE) -> str:
    """
    Construct the system + user prompt sent to the AI.
    """
    today = datetime.date.today().strftime("%B %d, %Y")

    # Format headlines as a numbered list
    headline_block = "\n".join(
        f"  {i+1}. [{h['topic']}] {h['title']}"
        for i, h in enumerate(headlines[:40])   # cap at 40 to stay within token limits
    )

    prompt = textwrap.dedent(f"""
        You are an expert technology journalist and SEO content strategist
        writing for the EdgePhone.ai blog on {today}.

        EdgePhone.ai (https://www.edgephone.ai/) builds the world's first
        smartphone designed entirely around on-device, privacy-first AI —
        zero cloud round-trips, full local inference, battery-efficient.

        ── RECENT NEWS HEADLINES (use for inspiration and context) ──
{headline_block}

        ── TASK ──
        Write exactly {num_articles} unique, SEO-optimised blog articles.
        Each article must:

        1. Be between 250–350 words.
        2. Have a compelling, keyword-rich title (max 80 chars).
        3. Contain a single paragraph excerpt (60–90 words) suitable for
           the <meta description> and card preview.
        4. Include 1–2 natural mentions of EdgePhone.ai and link back to
           {SITE_URL}.
        5. Target one of these SEO keyword themes per article (pick 4
           distinct ones across the set):
             Edge AI | On-device AI | Privacy-first AI |
             Lightweight AI models | Zero-latency AI |
             Mobile machine learning | Edge computing |
             On-device computer vision | Secure mobile AI
        6. Assign a short tag (e.g. "Edge AI", "Mobile ML", "Privacy AI",
           "Edge Computing").
        7. Be distinct — no two articles should cover the same angle.

        ── OUTPUT FORMAT ──
        Return a JSON array with exactly {num_articles} objects, each:
        {{
          "tag":      "<short tag>",
          "title":    "<article title>",
          "excerpt":  "<60-90 word excerpt>",
          "read_min": <integer minutes to read>,
          "body":     "<full article as HTML — use <p>, <h2>, <h3>, <ul>, <li>, <strong> tags. 300-400 words. Include 1-2 hyperlinks to https://www.edgephone.ai/ using <a href='https://www.edgephone.ai/'>EdgePhone.ai</a>.>"
        }}

        Return ONLY valid JSON — no markdown fences, no extra text.
    """).strip()

    return prompt


# ─────────────────────────────────────────────────────────────────
# STEP 3 – CALL AI API
# ─────────────────────────────────────────────────────────────────

def _detect_provider() -> str:
    """Return 'openai' or 'gemini' based on available env vars."""
    provider = os.environ.get("AI_PROVIDER", "").lower()
    if provider in ("openai", "gemini"):
        return provider
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    raise EnvironmentError(
        "No AI provider configured. "
        "Set OPENAI_API_KEY or GEMINI_API_KEY as environment / GitHub Actions secrets."
    )


def call_openai(prompt: str) -> str:
    """
    Call the OpenAI Chat Completions API.

    ► INSERT YOUR OPENAI API KEY as a GitHub Actions secret named OPENAI_API_KEY
    ► Optionally set AI_MODEL secret to override the model (default: gpt-4o)
    """
    import urllib.request, urllib.error

    api_key = os.environ["OPENAI_API_KEY"]
    model   = os.environ.get("AI_MODEL", DEFAULT_OPENAI_MODEL)

    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert technology journalist writing daily blog articles "
                    "for EdgePhone.ai. Always return pure JSON as instructed."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.75,
        "max_tokens": 4000,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    return body["choices"][0]["message"]["content"]


def call_gemini(prompt: str) -> str:
    """
    Call the Google Gemini (Generative Language) API.

    ► INSERT YOUR GEMINI API KEY as a GitHub Actions secret named GEMINI_API_KEY
    ► Optionally set AI_MODEL secret to override the model (default: gemini-1.5-pro)
    """
    import urllib.request

    api_key = os.environ["GEMINI_API_KEY"]
    model   = os.environ.get("AI_MODEL", DEFAULT_GEMINI_MODEL)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":    0.75,
            "maxOutputTokens": 4000,
        },
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    return body["candidates"][0]["content"]["parts"][0]["text"]


def generate_articles(headlines: list[dict]) -> list[dict]:
    """
    Orchestrate the AI call and return a list of article dicts.
    Falls back to stub articles if the API call fails (so CI never breaks).
    """
    prompt   = build_prompt(headlines)
    provider = _detect_provider()
    log.info("Using AI provider: %s", provider)

    raw_text = ""
    try:
        if provider == "openai":
            raw_text = call_openai(prompt)
        else:
            raw_text = call_gemini(prompt)

        # Strip accidental markdown fences
        clean = re.sub(r"^```[a-z]*\n?", "", raw_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\n?```$", "", clean.strip(), flags=re.MULTILINE)
        articles = json.loads(clean)

        if not isinstance(articles, list) or len(articles) == 0:
            raise ValueError("AI returned an empty or non-list response.")

        # Normalise: ensure exactly ARTICLES_TO_GENERATE entries
        articles = articles[:ARTICLES_TO_GENERATE]
        while len(articles) < ARTICLES_TO_GENERATE:
            articles.append(_stub_article(len(articles) + 1))

        log.info("Successfully generated %d articles.", len(articles))
        return articles

    except Exception as exc:
        log.error("AI generation failed: %s", exc)
        log.error("Raw API response was: %s", raw_text[:500])
        log.warning("Falling back to stub articles.")
        return [_stub_article(i + 1) for i in range(ARTICLES_TO_GENERATE)]


def _stub_article(n: int) -> dict:
    """Return a placeholder article when the AI call fails."""
    topics = [
        ("Edge AI",     "The Rise of Edge AI: Why Your Next Phone Won't Need the Cloud"),
        ("Mobile ML",   "Mobile Machine Learning in 2026: Smaller Models, Bigger Impact"),
        ("Privacy AI",  "Privacy-by-Default: How On-Device Inference Protects Your Data"),
        ("Edge Computing", "Edge Computing and IoT: Converging for a Zero-Latency Future"),
    ]
    tag, title = topics[(n - 1) % len(topics)]
    return {
        "tag": tag,
        "title": title,
        "excerpt": (
            f"EdgePhone.ai is pioneering a new era of on-device AI where every "
            f"computation stays on your device — private, instant, and independent "
            f"of cloud connectivity. Learn why this matters for the future of mobile "
            f"intelligence and how EdgePhone.ai leads the charge."
        ),
        "read_min": 5,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 4 – SLUG HELPER & RENDER ARTICLE HTML SNIPPETS
# ─────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert a title string to a URL-safe slug, e.g. 'Hello World!' → 'hello-world'."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)   # remove non-word chars (keep hyphens)
    text = re.sub(r"[\s_]+", "-", text)     # spaces/underscores → hyphens
    text = re.sub(r"-+", "-", text)         # collapse multiple hyphens
    return text.strip("-")[:80]             # cap length


def render_today_article(article: dict, pub_date: str) -> str:
    """
    Convert an article dict into a <article class="card"> HTML block.
    'Read more' links to article.html?id=<slug> instead of the main site.
    """
    tag      = html.escape(article.get("tag",   "Edge AI"))
    title    = html.escape(article.get("title", "Untitled"))
    excerpt  = html.escape(article.get("excerpt", ""))
    read_min = int(article.get("read_min", 5))
    slug     = article.get("id") or slugify(article.get("title", "article"))
    detail_url = f"article.html?id={slug}"

    return textwrap.dedent(f"""\
          <!-- ARTICLE -->
          <article class="card" itemscope itemtype="https://schema.org/BlogPosting">
            <div class="card-tag">{tag}</div>
            <h2 class="card-title" itemprop="headline">
              <a href="{detail_url}" itemprop="url">
                {title}
              </a>
            </h2>
            <p class="card-meta">
              <time itemprop="datePublished" datetime="{pub_date}">{_fmt_date(pub_date)}</time>
              &middot; <span>{read_min} min read</span>
            </p>
            <p class="card-excerpt" itemprop="description">
              {excerpt}
            </p>
            <a class="card-link" href="{detail_url}" aria-label="Read more about {slug}">
              Read more &rarr;
            </a>
          </article>""")


def render_archive_article(card_html: str) -> str:
    """
    Convert a full <article class="card"> block into the compact
    <article class="archive-card"> format. Preserves the article.html link.
    """
    tag_m     = re.search(r'class="card-tag"[^>]*>([^<]+)<', card_html)
    title_m   = re.search(r'itemprop="headline"[^>]*>\s*<a[^>]*>\s*(.*?)\s*</a>', card_html, re.DOTALL)
    date_m    = re.search(r'datetime="([^"]+)"', card_html)
    excerpt_m = re.search(r'itemprop="description"[^>]*>\s*(.*?)\s*</p>', card_html, re.DOTALL)
    # Preserve the original detail link if present
    link_m    = re.search(r'<a[^>]+href="(article\.html\?id=[^"]+)"', card_html)

    tag     = tag_m.group(1).strip()                                      if tag_m     else "Archive"
    title   = re.sub(r"\s+", " ", title_m.group(1)).strip()               if title_m   else "Archived Article"
    pub_iso = date_m.group(1)                                              if date_m    else ""
    excerpt = re.sub(r"\s+", " ", excerpt_m.group(1)).strip()             if excerpt_m else ""
    link    = link_m.group(1)                                              if link_m    else SITE_URL
    slug    = re.sub(r"[^a-z0-9 ]", "", title.lower())[:50].strip()

    return textwrap.dedent(f"""\
          <article class="archive-card" itemscope itemtype="https://schema.org/BlogPosting">
            <div>
              <p class="archive-card-tag">{tag}</p>
              <h3 class="archive-card-title" itemprop="headline">
                <a href="{link}" itemprop="url"
                   aria-label="Read archived article: {slug}">
                  {title}
                </a>
              </h3>
              <p class="archive-card-excerpt" itemprop="description">{excerpt[:200]}{"…" if len(excerpt) > 200 else ""}</p>
            </div>
            <p class="archive-card-meta">
              <time itemprop="datePublished" datetime="{pub_iso}">{_fmt_date(pub_iso)}</time>
            </p>
          </article>""")


def _fmt_date(iso: str) -> str:
    """Format an ISO date string (YYYY-MM-DD) to 'Month DD, YYYY'."""
    try:
        d = datetime.date.fromisoformat(iso)
        return d.strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return iso


# ─────────────────────────────────────────────────────────────────
# STEP 5 – MANIPULATE index.html
# ─────────────────────────────────────────────────────────────────

# Sentinel comments that delimit the injectable regions in index.html
TODAY_START  = "<!-- ── ARTICLES START (auto-updated by generate_newsletter.py) ── -->"
TODAY_END    = "<!-- ── ARTICLES END ── -->"
ARCHIVE_START = "<!-- ── ARCHIVE START (auto-updated by generate_newsletter.py) ── -->"
ARCHIVE_END   = "<!-- ── ARCHIVE END ── -->"


def _extract_between(content: str, start_marker: str, end_marker: str) -> str:
    """Return the text between two sentinel markers (exclusive)."""
    start_idx = content.find(start_marker)
    end_idx   = content.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return ""
    return content[start_idx + len(start_marker):end_idx]


def _replace_between(content: str, start_marker: str, end_marker: str, new_content: str) -> str:
    """Replace everything between two sentinel markers with new_content."""
    start_idx = content.find(start_marker)
    end_idx   = content.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        raise ValueError(
            f"Sentinel markers not found in HTML.\n"
            f"Expected: '{start_marker}' and '{end_marker}'"
        )
    return (
        content[: start_idx + len(start_marker)]
        + new_content
        + content[end_idx:]
    )


def extract_today_articles(html_content: str) -> list[str]:
    """
    Parse the current 'Today's Blogs' region and return individual
    <article ...> blocks as a list of strings.
    """
    region = _extract_between(html_content, TODAY_START, TODAY_END)
    # Split on each opening <article tag (preserving the tag itself)
    parts = re.split(r"(?=<article\s)", region, flags=re.IGNORECASE)
    articles = []
    for part in parts:
        stripped = part.strip()
        if stripped.lower().startswith("<article"):
            articles.append(stripped)
    return articles


def update_html(articles: list[dict]) -> None:
    """
    Main mutation function:
      1. Read index.html.
      2. Extract the existing Today articles.
      3. Convert them to archive format and prepend to the Archive section.
      4. Inject the new articles into Today's Blogs.
      5. Write the file back.
    """
    today_iso = datetime.date.today().isoformat()  # e.g. "2026-04-20"

    # ── Read ──────────────────────────────────────────────────────
    log.info("Reading %s", HTML_FILE)
    with open(HTML_FILE, encoding="utf-8") as fh:
        content = fh.read()

    # ── Harvest existing Today articles ──────────────────────────
    existing_articles = extract_today_articles(content)
    log.info("Found %d existing article(s) to archive.", len(existing_articles))

    # ── Convert existing Today → Archive format ───────────────────
    new_archive_entries = "\n".join(
        render_archive_article(a) for a in existing_articles
    )

    # ── Build the new Today article blocks ───────────────────────
    new_today_blocks = "\n".join(
        render_today_article(a, today_iso) for a in articles
    )

    # Wrap in blank lines for readability
    today_region   = f"\n{new_today_blocks}\n\n        "
    archive_region = f"\n{new_archive_entries}\n\n        "

    # ── Handle the case where the archive had only the placeholder ─
    existing_archive = _extract_between(content, ARCHIVE_START, ARCHIVE_END)
    placeholder      = '<p class="archive-empty">'
    if placeholder in existing_archive:
        # Replace placeholder with new entries
        updated_archive = archive_region
    else:
        # Prepend new entries before the existing archive content
        updated_archive = f"\n{new_archive_entries}{existing_archive}"

    # ── Inject Today ──────────────────────────────────────────────
    content = _replace_between(content, TODAY_START, TODAY_END, today_region)

    # ── Inject Archive ────────────────────────────────────────────
    content = _replace_between(content, ARCHIVE_START, ARCHIVE_END, updated_archive)

    # ── Write ─────────────────────────────────────────────────────
    log.info("Writing updated HTML to %s", HTML_FILE)
    with open(HTML_FILE, "w", encoding="utf-8") as fh:
        fh.write(content)

    log.info("Done! index.html updated successfully.")


# ─────────────────────────────────────────────────────────────────
# STEP 6 – UPDATE articles.json
# ─────────────────────────────────────────────────────────────────

def update_articles_json(articles: list[dict], today_iso: str) -> None:
    """
    Prepend the new articles to articles.json so article.html can load them.
    Assigns a slug-based 'id' field to each new article if not already set.
    Keeps the file capped at 120 entries to avoid unbounded growth.
    """
    MAX_ENTRIES = 120

    # Assign ids and dates to new articles
    for a in articles:
        if not a.get("id"):
            a["id"] = slugify(a.get("title", "article"))
        if not a.get("date"):
            a["date"] = today_iso

    # Load existing articles
    existing: list[dict] = []
    if os.path.exists(ARTICLES_JSON):
        try:
            with open(ARTICLES_JSON, encoding="utf-8") as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read articles.json (%s) — starting fresh.", exc)

    # Prepend new articles, dedup by id
    existing_ids = {a.get("id") for a in existing}
    new_entries  = [a for a in articles if a.get("id") not in existing_ids]
    merged       = new_entries + existing

    # Cap
    merged = merged[:MAX_ENTRIES]

    with open(ARTICLES_JSON, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)

    log.info("articles.json updated — %d total entries.", len(merged))


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== EdgePhone.ai Blog Generator — %s ===", datetime.date.today())

    # 1. Gather headlines
    headlines = gather_headlines(NEWS_TOPICS, per_topic=HEADLINES_PER_TOPIC)

    # 2. Generate articles via AI
    articles = generate_articles(headlines)

    today_iso = datetime.date.today().isoformat()

    # Assign slugs before writing anything so all steps share the same ids
    for a in articles:
        if not a.get("id"):
            a["id"] = slugify(a.get("title", "article"))

    # 3. Update articles.json (must run before update_html so ids are set)
    update_articles_json(articles, today_iso)

    # 4. Update index.html
    update_html(articles)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
