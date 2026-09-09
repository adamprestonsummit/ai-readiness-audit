import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import subprocess
import tempfile
import os
import base64
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

# python-docx for building the editable one-pager Word file
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Mm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Summit · AI Visibility Audit",
    page_icon="🔍",
    layout="wide",
)

# ─── Summit brand colours ──────────────────────────────────────────────────────
SUMMIT_RED   = "#D93B1A"   # primary red
SUMMIT_DARK  = "#1A1A1A"   # near-black
SUMMIT_GREY  = "#6B6B6B"
SUMMIT_LIGHT = "#F5F4F2"   # off-white background
WHITE        = "#FFFFFF"

LOGO_PATH = os.path.join(os.path.dirname(__file__), "summit_logo.png")

# ─── Gemini setup ─────────────────────────────────────────────────────────────
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

# ─── AI bot access check ──────────────────────────────────────────────────────
# The AI ecosystem publishes user-agents for the bots that fetch web content
# on behalf of ChatGPT, Claude, Perplexity, Gemini and friends. Site owners
# can (deliberately or accidentally) block these via robots.txt or WAF rules,
# which stops their content ever appearing in AI answers. We check both.
#
# References:
#   OpenAI:       https://platform.openai.com/docs/bots
#   Anthropic:    https://support.anthropic.com/en/articles/8896518
#   Google:       https://developers.google.com/search/docs/crawling-indexing/overview-google-crawlers
#   Perplexity:   https://docs.perplexity.ai/guides/bots
#   Common Crawl: https://commoncrawl.org/ccbot
AI_BOTS = [
    # (user_agent, vendor, purpose)
    ("GPTBot",             "OpenAI",       "Training crawler for ChatGPT models"),
    ("OAI-SearchBot",      "OpenAI",       "Indexes pages for ChatGPT Search"),
    ("ChatGPT-User",       "OpenAI",       "On-demand fetch when a user asks ChatGPT about a URL"),
    ("ClaudeBot",          "Anthropic",    "Training crawler for Claude"),
    ("Claude-Web",         "Anthropic",    "On-demand fetch when a user asks Claude about a URL"),
    ("anthropic-ai",       "Anthropic",    "Legacy Anthropic crawler user-agent"),
    ("Google-Extended",    "Google",       "Opt-out control for Gemini / Vertex AI training"),
    ("PerplexityBot",      "Perplexity",   "Indexes pages so they can be cited in Perplexity answers"),
    ("Perplexity-User",    "Perplexity",   "On-demand fetch when a Perplexity user opens a link"),
    ("CCBot",              "Common Crawl", "Training corpus used by many LLMs including GPT and Claude"),
    ("Applebot-Extended",  "Apple",        "Opt-out control for Apple Intelligence training"),
    ("Meta-ExternalAgent", "Meta",         "Training crawler for Meta AI / Llama"),
    ("Amazonbot",          "Amazon",       "Powers Alexa and Amazon AI answers"),
    ("Bytespider",         "ByteDance",    "Training crawler for Doubao / TikTok AI"),
    ("MistralAI-User",     "Mistral",      "On-demand fetch for Le Chat"),
    ("DuckAssistBot",      "DuckDuckGo",   "Indexes for DuckAssist AI answers"),
]

# UA strings paired with the fake header we send when doing live checks.
# We send a bot-shaped UA and see whether the site returns 200 or a block.
_LIVE_UA_TEMPLATE = "Mozilla/5.0 (compatible; {bot}/1.0; +https://example.com/bot)"

# The bots that matter most for AI visibility right now — blocking any of
# these has a direct, measurable impact on whether the site can appear in
# ChatGPT, Claude, Perplexity or Gemini answers. Used to raise a red
# callout at the top of every output.
CRITICAL_AI_BOTS = {
    "GPTBot":          ("ChatGPT",    "OpenAI's training crawler"),
    "OAI-SearchBot":   ("ChatGPT Search", "Indexes pages for ChatGPT's search product"),
    "ClaudeBot":       ("Claude",     "Anthropic's training + citation crawler"),
    "PerplexityBot":   ("Perplexity", "Indexes pages so they can be cited in Perplexity answers"),
    "Google-Extended": ("Gemini",     "Opt-out control for Gemini training"),
}


def get_blocked_critical_bots(bot_access: dict) -> list[dict]:
    """
    Return the list of CRITICAL_AI_BOTS whose overall verdict is 'blocked'.
    Each entry is the full bot dict from bot_access['bots'] with an extra
    'product' and 'why_it_matters' pulled from CRITICAL_AI_BOTS.
    """
    if not bot_access or not isinstance(bot_access, dict):
        return []
    bots = bot_access.get("bots") or []
    out = []
    for b in bots:
        ua = b.get("user_agent", "")
        if ua in CRITICAL_AI_BOTS and b.get("verdict") == "blocked":
            product, why = CRITICAL_AI_BOTS[ua]
            enriched = dict(b)
            enriched["product"] = product
            enriched["why_it_matters"] = why
            out.append(enriched)
    return out


def _fetch_robots_txt(domain: str, session: requests.Session) -> tuple[str, str, str]:
    """
    Try https then http, with and without www, until we find a robots.txt.
    Returns (fetched_url, content, error_message). One of content/error is set.
    """
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    host = parsed.netloc or parsed.path
    host = host.strip("/")
    candidates = []
    for scheme in ("https", "http"):
        for h in ({host, host.removeprefix("www."), f"www.{host.removeprefix('www.')}"}):
            candidates.append(f"{scheme}://{h}/robots.txt")
    seen = set()
    last_err = ""
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        try:
            r = session.get(url, timeout=8, allow_redirects=True)
            if r.status_code == 200 and r.text.strip():
                return url, r.text, ""
            last_err = f"HTTP {r.status_code} at {url}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__} at {url}"
    return "", "", last_err or "robots.txt not found on any host/scheme variant"


def _robots_status_for_bot(robots_text: str, bot_ua: str) -> tuple[str, str]:
    """
    Determine what the robots.txt says about a specific bot user-agent.
    Returns (status, evidence) where status is one of:
      - "blocked_all"          Disallow: / for this UA (or a User-agent: * that matches)
      - "blocked_partial"      Some Disallow rules but not "/"
      - "allowed"              Explicit rules exist and none block "/"
      - "not_specified"        No explicit rules for this UA, and * has no Disallow
      - "no_robots"            No robots.txt found at all
    """
    if not robots_text:
        return "no_robots", "No robots.txt file found on the site."

    # urllib.robotparser handles the RFC properly (longest-match user-agent,
    # rule precedence, etc). We use it for the authoritative allow/deny call
    # on "/", then supplement with a manual scan for the "did this UA appear
    # at all?" question, since RobotFileParser hides that from us.
    try:
        rp = RobotFileParser()
        rp.parse(robots_text.splitlines())
        can_root = rp.can_fetch(bot_ua, "/")
    except Exception:
        can_root = True  # be charitable if the file is malformed

    # Manual scan for explicit mention & any Disallow rules
    ua_re = re.compile(r"^\s*user-agent\s*:\s*(.+?)\s*(?:#.*)?$", re.I | re.M)
    lines = robots_text.splitlines()
    explicit_group = False
    disallow_lines: list[str] = []
    in_matching_group = False
    for ln in lines:
        m = ua_re.match(ln)
        if m:
            ua = m.group(1).strip().lower()
            if ua == bot_ua.lower():
                in_matching_group = True
                explicit_group = True
            else:
                in_matching_group = False
            continue
        if in_matching_group:
            s = ln.strip()
            if s.lower().startswith("disallow:"):
                val = s.split(":", 1)[1].strip()
                if val:
                    disallow_lines.append(val)

    if not can_root:
        # Determine whether that block came from this bot's group or from *
        if explicit_group and any(d == "/" for d in disallow_lines):
            return "blocked_all", "Explicitly blocked from the whole site via robots.txt: 'User-agent: %s' + 'Disallow: /'." % bot_ua
        return "blocked_all", "Blocked from the site root by a robots.txt rule (likely a 'User-agent: *' catch-all with 'Disallow: /')."

    if explicit_group:
        if disallow_lines:
            return "blocked_partial", "Specific paths are disallowed for %s (e.g. %s) but the site root is still crawlable." % (bot_ua, ", ".join(disallow_lines[:3]))
        return "allowed", "%s is explicitly named in robots.txt and has no Disallow rules." % bot_ua

    return "not_specified", "robots.txt does not mention %s. Default behaviour applies (site is crawlable)." % bot_ua


def _live_check_bot(url: str, bot_ua: str, session: requests.Session) -> dict:
    """Fetch the URL as the given bot UA and record the response."""
    headers = {
        "User-Agent":      _LIVE_UA_TEMPLATE.format(bot=bot_ua),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    try:
        r = session.get(url, headers=headers, timeout=10, allow_redirects=True)
        code = r.status_code
        if code == 200:
            status = "allowed"
        elif code in (401, 403, 429, 451):
            status = "blocked"
        elif 500 <= code < 600:
            status = "server_error"
        else:
            status = "other"
        return {"live_status": status, "http_code": code, "live_error": ""}
    except requests.Timeout:
        return {"live_status": "timeout", "http_code": 0, "live_error": "Request timed out after 10s"}
    except requests.RequestException as e:
        return {"live_status": "error", "http_code": 0, "live_error": f"{type(e).__name__}"}


def check_ai_bot_access(domain: str, session: requests.Session | None = None) -> dict:
    """
    Runs the robots.txt + live-WAF check for every AI bot in AI_BOTS.
    Returns a dict safe to attach to the audit result and render in UI/docs.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        })

    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    homepage = f"{parsed.scheme or 'https'}://{parsed.netloc or parsed.path.strip('/')}/"

    robots_url, robots_text, robots_err = _fetch_robots_txt(domain, session)

    # Build the per-bot status list
    bots: list[dict] = []

    # Live checks run concurrently — one HTTP request per bot, capped at 6
    # workers to be gentle on the target host.
    live_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(_live_check_bot, homepage, bot_ua, session): bot_ua
            for bot_ua, _, _ in AI_BOTS
        }
        for fut in as_completed(futs):
            bot_ua = futs[fut]
            try:
                live_results[bot_ua] = fut.result()
            except Exception as e:
                live_results[bot_ua] = {
                    "live_status": "error", "http_code": 0,
                    "live_error": f"{type(e).__name__}: {e}",
                }

    for bot_ua, vendor, purpose in AI_BOTS:
        rob_status, rob_evidence = _robots_status_for_bot(robots_text, bot_ua)
        live = live_results.get(bot_ua, {"live_status": "unknown", "http_code": 0, "live_error": ""})
        # Overall access verdict combines the two signals
        if rob_status == "blocked_all" or live["live_status"] == "blocked":
            verdict = "blocked"
        elif rob_status in ("blocked_partial",) or live["live_status"] in ("timeout", "error", "server_error", "other"):
            verdict = "partial"
        else:
            verdict = "allowed"
        bots.append({
            "user_agent":       bot_ua,
            "vendor":           vendor,
            "purpose":          purpose,
            "robots_status":    rob_status,
            "robots_evidence":  rob_evidence,
            "live_status":      live["live_status"],
            "http_code":        live["http_code"],
            "live_error":       live["live_error"],
            "verdict":          verdict,
        })

    summary = {
        "allowed":  sum(1 for b in bots if b["verdict"] == "allowed"),
        "partial":  sum(1 for b in bots if b["verdict"] == "partial"),
        "blocked":  sum(1 for b in bots if b["verdict"] == "blocked"),
        "total":    len(bots),
    }

    return {
        "homepage_checked":    homepage,
        "robots_txt_url":      robots_url,
        "robots_txt_found":    bool(robots_text),
        "robots_txt_content":  robots_text[:8000],  # cap for storage
        "robots_txt_error":    robots_err,
        "bots":                bots,
        "summary":             summary,
    }


# ─── Fetch URL HTML ───────────────────────────────────────────────────────────
# Hard cap on raw HTML size before we even parse it. On extreme PLPs
# (e.g. category pages with 100+ products fully rendered inline) some
# sites return 5-10MB of HTML. Parsing that much can take 30+ seconds
# with BeautifulSoup and generates far more signal than Gemini needs.
# 3MB comfortably covers even the largest real-world pages.
MAX_HTML_BYTES = 3_000_000

def extract_page_signals(url: str, html: str) -> str:
    """
    Compress a raw HTML page into a compact signal summary for Gemini.
    Instead of sending thousands of tokens of raw HTML, we extract exactly
    what an AI auditor needs: meta, schema, headings, links, alt text, ARIA.
    This keeps each page under ~2000 tokens while preserving all audit signals.
    """
    import re as _re
    original_html_len = len(html)
    html_was_capped = False
    if original_html_len > MAX_HTML_BYTES:
        # Keep the head (all meta/schema is up top) plus a slice of the body
        # so link/alt/heading extraction still works on the visible portion.
        html = html[:MAX_HTML_BYTES]
        html_was_capped = True
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "path"]):
        tag.decompose()

    out = [f"URL: {url}"]
    if html_was_capped:
        out.append(
            f"[EXTRACTOR_HTML_CAPPED: page was {original_html_len} bytes, "
            f"processed the first {MAX_HTML_BYTES} bytes only. The head "
            f"(meta/schema) is fully covered but body-derived signals "
            f"(headings, links, alt, body content) reflect only the "
            f"beginning of the page. Do not treat this as a site issue.]"
        )

    # ── Meta & Open Graph ──────────────────────────────────────────────
    meta_items = []
    title = soup.find("title")
    if title: meta_items.append(f"title: {title.get_text().strip()[:120]}")
    for m in soup.find_all("meta"):
        name = m.get("name","") or m.get("property","")
        val  = m.get("content","")
        if name and val and name.lower() in [
            "description","robots","author","article:author",
            "article:published_time","article:modified_time",
            "og:title","og:type","og:description","og:url","og:image",
            "og:locale","og:site_name","twitter:card","twitter:title",
            "twitter:description","viewport"
        ]:
            meta_items.append(f"{name}: {val[:120]}")
    link_tags = []
    for l in soup.find_all("link", rel=True):
        rel = " ".join(l.get("rel",[]))
        if rel in ["canonical","alternate"]:
            link_tags.append(f"<link rel={rel} href={l.get('href','')[:80]}>")
    out.append("META:\n" + "\n".join(meta_items[:20] + link_tags[:5]))

    # ── Schema / JSON-LD ──────────────────────────────────────────────
    # Instead of sending truncated raw JSON (which makes Gemini incorrectly
    # flag pages as having "truncated schema"), we summarise each schema
    # block: type, key fields, and validity. This works even for PLPs with
    # dozens of Product schemas — each is summarised, not truncated.
    import json as _json

    def _typestr(t) -> str:
        """
        @type in schema.org can be a string, a list of strings, or a URL.
        Normalise all cases to a compact display string.
        """
        if t is None:
            return "?"
        if isinstance(t, list):
            parts = []
            for x in t:
                if isinstance(x, str):
                    parts.append(x.rstrip("/").split("/")[-1] or x)
                else:
                    parts.append("?")
            return "+".join(parts) if parts else "?"
        if isinstance(t, str):
            return t.rstrip("/").split("/")[-1] or t
        return "?"

    def _schema_summary(obj, depth=0):
        """Return a short human summary of a schema.org object."""
        if depth > 3:
            return "..."
        if isinstance(obj, list):
            # ItemList / array — summarise count and first item's type
            if not obj:
                return "empty list"
            first_type = "?"
            if isinstance(obj[0], dict):
                first_type = _typestr(obj[0].get("@type"))
            return f"list of {len(obj)} items (first @type={first_type})"
        if not isinstance(obj, dict):
            return type(obj).__name__
        typ = _typestr(obj.get("@type"))
        # Handle @graph — array of nested schemas
        if "@graph" in obj and isinstance(obj["@graph"], list):
            types_in_graph = [
                _typestr(g.get("@type")) if isinstance(g, dict) else "?"
                for g in obj["@graph"]
            ]
            return f"@graph with {len(obj['@graph'])} nodes: {', '.join(types_in_graph[:10])}"
        # Extract useful fields based on type
        # Fields where we expand nested sub-fields so Gemini can see what is
        # actually populated, instead of just the nested @type (which hides
        # completeness and causes false "incomplete" flags).
        # Fields where we expand nested sub-fields so Gemini can see what is
        # actually populated. Only REQUIRED sub-fields are ever reported as
        # "missing" — optional ones are simply omitted when absent, so the
        # word "missing" only appears for genuine gaps, not for fields most
        # sites never populate (e.g. priceValidUntil, author url, publisher logo).
        NESTED_FIELDS_REQUIRED = {
            "address":   ["streetAddress","addressLocality","postalCode","addressCountry"],
            "offers":    ["price","priceCurrency","availability"],
            "author":    ["name"],
            "publisher": ["name"],
        }
        NESTED_FIELDS_OPTIONAL = {
            "address":   ["addressRegion"],
            "offers":    ["url","priceValidUntil"],
            "author":    ["url"],
            "publisher": ["url","logo"],
        }

        useful = []
        for k in ["name","headline","url","description","datePublished","dateModified",
                  "author","publisher","offers","aggregateRating","brand","sku",
                  "image","logo","email","telephone","address","itemListElement"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, (dict, list)):
                    if k == "itemListElement" and isinstance(v, list):
                        useful.append(f"{k}=[{len(v)} items]")
                    elif isinstance(v, dict) and k in NESTED_FIELDS_REQUIRED:
                        required = NESTED_FIELDS_REQUIRED[k]
                        optional = NESTED_FIELDS_OPTIONAL.get(k, [])
                        present_req = [sf for sf in required if v.get(sf)]
                        missing_req = [sf for sf in required if not v.get(sf)]
                        present_opt = [sf for sf in optional if v.get(sf)]
                        nested_type = _typestr(v.get("@type"))
                        all_present = present_req + present_opt
                        detail = f"present: {', '.join(all_present) if all_present else 'none'}"
                        if missing_req:
                            detail += f" | missing required: {', '.join(missing_req)}"
                        useful.append(f"{k}={{{nested_type}: {detail}}}")
                    elif isinstance(v, dict):
                        useful.append(f"{k}={{{_typestr(v.get('@type'))}}}")
                    else:
                        useful.append(f"{k}=[{len(v)}]")
                else:
                    val = str(v)[:60]
                    useful.append(f"{k}='{val}'")
        return f"@type={typ}" + ((" " + ", ".join(useful)) if useful else "")

    def _parse_json_loose(txt):
        """Try to parse JSON, stripping common issues (comments, trailing commas)."""
        txt = txt.strip()
        try:
            return _json.loads(txt)
        except _json.JSONDecodeError:
            # Try stripping trailing commas
            cleaned = _re.sub(r",\s*([}\]])", r"\1", txt)
            try:
                return _json.loads(cleaned)
            except _json.JSONDecodeError:
                return None

    schemas = []
    schema_scripts = list(soup.find_all("script", attrs={"type": "application/ld+json"}))
    # Fallback: raw HTML regex if BeautifulSoup missed any
    raw_blocks = []
    if not schema_scripts:
        raw_blocks = _re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, _re.DOTALL | _re.IGNORECASE
        )

    schema_sources = [(s.string or s.get_text()) for s in schema_scripts] or raw_blocks
    total_blocks = len(schema_sources)

    for i, raw_txt in enumerate(schema_sources):
        if not raw_txt or not raw_txt.strip():
            continue
        parsed = _parse_json_loose(raw_txt)
        if parsed is None:
            # Note the parse error — this IS a real signal Gemini should see
            schemas.append(f"BLOCK {i+1}: PARSE_ERROR (invalid JSON, {len(raw_txt)} chars)")
        else:
            if isinstance(parsed, list):
                summary = _schema_summary(parsed)
                schemas.append(f"BLOCK {i+1}: {summary}")
            else:
                summary = _schema_summary(parsed)
                schemas.append(f"BLOCK {i+1}: {summary}")

    if schemas:
        out.append(
            f"SCHEMA_JSON_LD ({total_blocks} block(s) found, all summarised — NOT truncated):\n"
            + "\n".join(schemas)
        )
    else:
        out.append("SCHEMA_JSON_LD: NONE FOUND")

    # ── Pre-labelled dates ────────────────────────────────────────────
    # Extract every date-like value from schema markup and meta tags,
    # then label each as PAST or FUTURE against today's real date.
    # This means Gemini never has to decide "is 2026 in the future?".
    from datetime import date as _date
    today = _date.today()
    date_findings = []
    # Combine all schema JSON-LD content + full raw HTML for date scanning
    scan_text = " ".join(schemas) + " " + html
    # ISO-style dates like 2026-06-02 or 2026-06-02T12:34:56
    iso_pattern = _re.compile(r'\b(20\d{2})-(\d{2})-(\d{2})(?:T\d{2}:\d{2}[\d:.+\-Z]*)?\b')
    seen = set()
    for m in iso_pattern.finditer(scan_text):
        raw_date = m.group(0)
        if raw_date in seen:
            continue
        seen.add(raw_date)
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            found = _date(y, mo, d)
            label = "PAST" if found <= today else "FUTURE"
            # Find nearby context (60 chars before) to help identify what field this is
            start = max(0, m.start() - 60)
            context = scan_text[start:m.start()].strip()[-40:]
            date_findings.append(f"{raw_date} [{label}] ...{context}")
        except ValueError:
            continue
        if len(date_findings) >= 15:
            break

    if date_findings:
        out.append(
            f"DATES_FOUND (today is {today.isoformat()}):\n" + "\n".join(date_findings)
        )
    else:
        out.append(f"DATES_FOUND (today is {today.isoformat()}): NONE FOUND")

    # ── Heading structure ─────────────────────────────────────────────
    # Collect headings by level so H1s are never crowded out by nav H2/H3s.
    # Also include an H1 count so Gemini can spot multiple-H1 problems.
    heads_by_level = {"h1": [], "h2": [], "h3": [], "h4": []}
    for tag in soup.find_all(["h1","h2","h3","h4"]):
        txt = tag.get_text(" ", strip=True)[:100]
        if txt and tag.name in heads_by_level:
            heads_by_level[tag.name].append(txt)

    # Fallback: some CMSs use role="heading" with aria-level instead of real h1-h6
    if not heads_by_level["h1"]:
        for tag in soup.find_all(attrs={"role":"heading"}):
            level = tag.get("aria-level","1")
            txt = tag.get_text(" ", strip=True)[:100]
            if txt and str(level) == "1":
                heads_by_level["h1"].append(txt + " [role=heading aria-level=1]")

    h1_count = len(heads_by_level["h1"])
    heading_lines = [f"H1 COUNT: {h1_count}"]
    for lvl in ["h1","h2","h3","h4"]:
        # Always include ALL h1s; cap h2/h3/h4 at 8 each so we don't blow the budget
        cap = None if lvl == "h1" else 8
        items = heads_by_level[lvl][:cap] if cap else heads_by_level[lvl]
        for txt in items:
            heading_lines.append(f"<{lvl}>{txt}</{lvl}>")
    out.append("HEADINGS:\n" + ("\n".join(heading_lines) if h1_count or any(heads_by_level.values()) else "NONE FOUND"))

    # ── ARIA usage ────────────────────────────────────────────────────
    aria_items = []
    for tag in soup.find_all(True):
        role  = tag.get("role","")
        label = tag.get("aria-label","") or tag.get("aria-labelledby","")
        curr  = tag.get("aria-current","")
        if role:   aria_items.append(f"role={role}")
        if label:  aria_items.append(f"aria-label={label[:60]}")
        if curr:   aria_items.append(f"aria-current={curr}")
        if len(aria_items) >= 15: break
    out.append("ARIA: " + ("; ".join(aria_items) if aria_items else "NONE FOUND"))

    # ── Navigation / links sample ─────────────────────────────────────
    nav = soup.find("nav")
    all_links = soup.find_all("a", href=True)
    link_sample = []
    for a in all_links[:40]:
        href = a.get("href","")
        txt  = a.get_text(" ",strip=True)[:50]
        link_sample.append(f"{txt} -> {href[:80]}")
    out.append(f"LINKS (total found: {len(all_links)}):\n" + "\n".join(link_sample[:25]))

    # ── Images / alt text ─────────────────────────────────────────────
    imgs = soup.find_all("img")
    img_sample = []
    for img in imgs[:20]:
        alt = img.get("alt", "MISSING")
        src = img.get("src","")[:60]
        img_sample.append(f"alt={repr(alt)} src={src}")
    out.append(f"IMAGES (total: {len(imgs)}):\n" + "\n".join(img_sample))

    # ── Inline JS data stores (Next.js, Nuxt, etc.) ─────────────────
    # Even JS-heavy sites embed page data in __NEXT_DATA__ or similar
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            import json as _json
            nd = _json.loads(next_data.string)
            # Extract text-like values up to 600 chars
            def _extract_strings(obj, depth=0):
                if depth > 4: return []
                if isinstance(obj, str) and len(obj) > 20:
                    return [obj[:120]]
                if isinstance(obj, dict):
                    vals = []
                    for v in obj.values():
                        vals.extend(_extract_strings(v, depth+1))
                    return vals[:10]
                if isinstance(obj, list):
                    vals = []
                    for v in obj[:5]:
                        vals.extend(_extract_strings(v, depth+1))
                    return vals[:10]
                return []
            strings = _extract_strings(nd)
            if strings:
                out.append("NEXT_DATA_CONTENT:\n" + "\n".join(strings[:15]))
        except Exception:
            pass

    # ── Body content sample (for LLM & CONTENT signal) ───────────────
    # Priority order: prefer <main>, then <article>, then a #content /
    # role="main" container, then finally fall back to <body> minus the
    # nav/header/footer. This ensures we sample the ACTUAL page content
    # (product descriptions, article body, etc.) not the mega-menu.
    body_sample = ""
    main_candidates = []

    # Try semantic containers first
    for selector in [
        {"name": "main"},
        {"name": "article"},
        {"attrs": {"role": "main"}},
        {"attrs": {"id": _re.compile(r"main|content|product", _re.I)}},
        {"attrs": {"class": _re.compile(r"product-description|product-details|product__description|main-content|page-content|article-body|entry-content|post-content", _re.I)}},
    ]:
        try:
            found = soup.find_all(**selector)
        except Exception:
            found = []
        for f in found:
            txt = " ".join(f.get_text(" ", strip=True).split())
            if len(txt) > 100:
                main_candidates.append(txt)
        if main_candidates:
            break

    if main_candidates:
        # Use the longest main candidate — biggest content block wins
        main_candidates.sort(key=len, reverse=True)
        body_sample = main_candidates[0][:3500]
        out.append(f"MAIN_CONTENT (from semantic container):\n{body_sample}")
    else:
        # Fallback: full body minus nav/header/footer/aside
        body = soup.find("body")
        if body:
            body_copy = BeautifulSoup(str(body), "html.parser")
            for junk in body_copy(["nav","header","footer","aside","form"]):
                junk.decompose()
            # Also strip common nav/footer class patterns
            for junk in body_copy.find_all(
                attrs={"class": _re.compile(r"nav|menu|header|footer|cookie|banner|breadcrumb|sidebar", _re.I)}
            ):
                junk.decompose()
            text = " ".join(body_copy.get_text(" ", strip=True).split())[:3500]
            out.append(f"BODY_TEXT_SAMPLE (nav stripped):\n{text}")

    # Also include a shorter raw body sample so Gemini can still audit
    # nav content signals like "Shop by Category" etc.
    body = soup.find("body")
    if body:
        raw_snippet = " ".join(body.get_text(" ", strip=True).split())[:800]
        out.append(f"RAW_BODY_START (first 800 chars including nav):\n{raw_snippet}")

    return "\n\n".join(out)


def _detect_block_or_empty(status: int, html: str, url: str) -> tuple[bool, str]:
    """
    Combined block and emptiness detector. Returns (is_bad, reason).
    Runs BEFORE any signal extraction. If this returns True, we do not score,
    we do not send to Gemini, we surface an error and prompt the user to paste
    raw HTML instead.
    """
    # Signal 1: HTTP status
    if status in (401, 403, 429, 503):
        return True, f"HTTP {status} response (likely bot block or rate limit)"

    # Signal 2: response too small to be a real page
    if len(html) < 2000:
        return True, f"Response body only {len(html)} bytes (likely a block or error page)"

    # Signal 3: known WAF and bot-challenge fingerprints
    snippet = html[:5000].lower()
    fingerprints = {
        "just a moment":              "Cloudflare bot challenge",
        "_cf_chl_":                   "Cloudflare bot challenge",
        "cf-browser-verification":    "Cloudflare bot challenge",
        "cf_chl_opt":                 "Cloudflare bot challenge",
        "checking your browser":      "Cloudflare bot challenge",
        "attention required":         "Cloudflare block",
        "ray id":                     "Cloudflare block page",
        "access denied":              "Access denied response",
        "403 forbidden":              "403 Forbidden response",
        "enable javascript and cookies to continue": "Bot challenge (JS+cookies required)",
        "ak-bmsc":                    "Akamai bot manager challenge",
        "reference #18.":             "Akamai bot manager challenge",
        "px-captcha":                 "PerimeterX challenge",
        "dd-protection":              "DataDome challenge",
        "incapsula":                  "Imperva Incapsula challenge",
        "distil_r_captcha":           "Distil Networks challenge",
    }
    for marker, description in fingerprints.items():
        if marker in snippet:
            return True, description

    # Signal 4: title heuristics
    _soup = BeautifulSoup(html, "html.parser")
    _title_tag = _soup.find("title")
    if _title_tag and _title_tag.string:
        _title = _title_tag.string.strip().lower()
        block_titles = [
            "just a moment", "access denied", "attention required",
            "please wait", "403 forbidden", "you have been blocked",
            "security check", "checking your browser",
        ]
        if any(bt in _title for bt in block_titles):
            return True, f"Blocking page title: {_title_tag.string.strip()[:80]}"

    # Signal 5: structural emptiness
    # A real page has headings, links, and paragraphs. Challenge pages do not.
    # This catches blocks even when no fingerprint matches, which is what
    # went wrong on Bettys.
    for t in _soup(["script", "style", "noscript"]):
        t.decompose()
    body = _soup.find("body")
    if body:
        h1_count   = len(body.find_all("h1"))
        h_count    = len(body.find_all(["h1", "h2", "h3"]))
        link_count = len(body.find_all("a", href=True))
        p_count    = len(body.find_all("p"))
        body_text  = body.get_text(" ", strip=True)
        if h_count == 0 and link_count < 3 and p_count < 3:
            return True, (
                f"Page appears empty (h1={h1_count}, headings={h_count}, "
                f"links={link_count}, paragraphs={p_count}, body chars={len(body_text)}). "
                "Likely a bot-block, WAF challenge, or JS-only shell."
            )

    return False, ""


def _is_js_shell(html: str) -> bool:
    """Return True if page has almost no body text but many script tags."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript","svg"]): t.decompose()
    body_text = soup.get_text(" ", strip=True)
    script_count = html.lower().count("<script")
    return len(body_text) < 300 and script_count > 3


def _fetch_direct(url: str, session: requests.Session) -> tuple:
    """Returns (html, status, source_label)."""
    r = session.get(url, timeout=15, allow_redirects=True)
    return r.text, r.status_code, "direct"


def _fetch_wayback(url: str, session: requests.Session) -> tuple:
    """
    Fetch via Wayback Machine (web.archive.org).
    Wayback crawls with Googlebot, so it bypasses most bot-protection
    and returns what search/AI crawlers actually saw.
    Returns (html, status, source_label) or raises on failure.
    """
    # Step 1: find the latest snapshot URL
    avail_api = f"https://archive.org/wayback/available?url={url}"
    meta = session.get(avail_api, timeout=10).json()
    snapshots = meta.get("archived_snapshots", {})
    closest   = snapshots.get("closest", {})
    snap_url  = closest.get("url", "")
    if not snap_url:
        raise ValueError("No Wayback snapshot found")
    # Convert to raw snapshot (remove Wayback toolbar injection)
    # id_ suffix returns the original page without Wayback toolbar
    snap_url = snap_url.replace("/web/", "/web/") 
    raw_url  = snap_url.replace("http://web.archive.org/web/",
                                 "https://web.archive.org/web/")
    # Insert 'id_' flag to get clean original HTML
    parts = raw_url.split("/web/", 1)
    if len(parts) == 2:
        ts_and_url = parts[1].split("/", 1)
        if len(ts_and_url) == 2:
            raw_url = f"https://web.archive.org/web/{ts_and_url[0]}id_/{ts_and_url[1]}"
    r = session.get(raw_url, timeout=20, allow_redirects=True)
    return r.text, r.status_code, f"Wayback Machine ({closest.get('timestamp','')})"


def fetch_single_page(url: str, session: requests.Session) -> tuple:
    """
    Multi-strategy fetcher. Returns (signals_text, fetch_note).
    Strategy: direct, then Wayback fallback. If both fail or return
    blocked/empty content, returns a FETCH_BLOCKED marker that the
    orchestrator surfaces to the user with a prompt to paste raw HTML.
    """
    fetch_note = ""
    html = ""
    label = ""
    block_reason = ""
    direct_error = ""

    # Strategy 1: direct fetch
    try:
        html, status, label = _fetch_direct(url, session)
        is_bad, reason = _detect_block_or_empty(status, html, url)
        if is_bad:
            block_reason = reason
            raise ValueError(reason)
    except Exception as e:
        direct_error = str(e)
        # Strategy 2: Wayback Machine
        try:
            html, status, label = _fetch_wayback(url, session)
            is_bad, reason = _detect_block_or_empty(status, html, url)
            if is_bad:
                # Wayback also returned junk. Give up and surface the block.
                return (
                    f"URL: {url}\n\n[FETCH_BLOCKED: {block_reason or direct_error}. "
                    f"Wayback fallback also returned unusable content: {reason}. "
                    "Paste the raw HTML for this URL using the 'Site blocked?' "
                    "expander below the URL inputs.]",
                    "blocked"
                )
            fetch_note = (
                f"SOURCE_NOTE: Direct fetch was blocked ({block_reason or direct_error}). "
                f"Content retrieved from {label}, reflects how AI crawlers "
                "that use cached versions see this page."
            )
        except Exception as e2:
            return (
                f"URL: {url}\n\n[FETCH_BLOCKED: Direct fetch failed ({direct_error}). "
                f"Wayback fetch also failed ({e2}). "
                "The site is either heavily bot-protected or unreachable. "
                "Paste the raw HTML for this URL using the 'Site blocked?' "
                "expander below the URL inputs.]",
                "blocked"
            )

    # Detect JS shell (still useful for legitimate React/Next/Vue sites)
    is_shell = _is_js_shell(html)
    signals  = extract_page_signals(url, html)

    if is_shell:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style","noscript","svg"]): t.decompose()
        body_len = len(soup.get_text(" ", strip=True))
        sc       = html.lower().count("<script")
        fetch_note += (
            f"\nJS_SHELL_NOTE: Page appears client-side rendered "
            f"(body text {body_len} chars, {sc} script tags). "
            "Score CRAWL 1-3. Other dimensions scored from shell content above."
        )

    if fetch_note:
        signals += "\n\n" + fetch_note

    return signals, label


def _normalise_url(u: str) -> str:
    """
    Normalise a URL for comparison/deduping.
    Handles: scheme (http/https), www prefix, trailing slash, casing of host.
    Preserves path casing (paths ARE case-sensitive).
    """
    if not u:
        return ""
    u = u.strip()
    # Ensure scheme
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    from urllib.parse import urlparse, urlunparse
    try:
        p = urlparse(u)
        host = p.netloc.lower()
        # Strip www. for comparison purposes
        if host.startswith("www."):
            host = host[4:]
        # Force https scheme for comparison
        path = p.path.rstrip("/") or "/"
        return f"https://{host}{path}"
    except Exception:
        return u.rstrip("/").lower()


def fetch_pages(domain: str, extra_urls: list[str],
                pasted_html: dict | None = None) -> dict[str, str]:
    """
    Fetch up to 4 pages. pasted_html is an optional {url: html} dict
    for manually pasted content (bypasses fetch entirely for those URLs).

    Key rules:
    - When pasted_html has content, PASTED URLs are always preferred over
      network fetches, even if the network could succeed.
    - When ANY URL matches a pasted entry (after normalising scheme/www/
      trailing-slash differences), that pasted HTML is used.
    - Homepage is only added to the list if there's room after pasted URLs
      and it hasn't already been provided via pasted_html.
    """
    base = domain.rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base

    # Build a fast-lookup index of pasted URLs by normalised form
    pasted_index = {}   # normalised_url -> (original_url, html)
    for pu, ph in (pasted_html or {}).items():
        key = _normalise_url(pu)
        if key:
            pasted_index[key] = (pu, ph)

    # Build the master URL list
    seen = set()
    ordered_urls = []

    def add_url(u):
        norm = _normalise_url(u)
        if norm and norm not in seen:
            seen.add(norm)
            ordered_urls.append(u.strip().rstrip("/"))

    # 1. Pasted URLs first (guaranteed slots)
    for pu in (pasted_html or {}).keys():
        add_url(pu)

    # 2. Homepage — only if not already covered by a pasted URL
    if _normalise_url(base) not in seen:
        add_url(base)

    # 3. Extra typed URLs
    for u in extra_urls:
        add_url(u)

    # Cap at 4
    urls = ordered_urls[:4]

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Cache-Control":   "no-cache",
    })

    pages = {}
    for url in urls:
        norm = _normalise_url(url)
        pasted_match = pasted_index.get(norm)

        if pasted_match:
            _orig_url, ph = pasted_match
            signals = extract_page_signals(url, ph)
            signals += "\n\nSOURCE_NOTE: HTML was manually provided."
            pages[url] = signals
        else:
            signals, _label = fetch_single_page(url, session)
            # SAFETY NET: if fetch failed/blocked AND we have any pasted HTML
            # at all, note the mismatch so the UI can help the user.
            if ("FETCH_BLOCKED" in signals or "FETCH_FAILED" in signals) and pasted_html:
                signals += (
                    f"\n\nFETCH_HINT: You pasted HTML for other URLs but this "
                    f"URL ({url}) was not matched. Check that the URL in the "
                    f"'Site blocked by bot protection' paste form matches "
                    f"exactly (scheme, www, trailing slash all handled)."
                )
            pages[url] = signals

    return pages

# ─── Gemini audit ─────────────────────────────────────────────────────────────
AUDIT_PROMPT = """
You are an expert AI visibility auditor. Audit the provided HTML pages exactly like a senior technical SEO and AI readiness consultant would.

DATE CONTEXT (READ FIRST):
The value {TODAY_DATE} at the end of the "Pages to audit" section is the current real-world date the audit is being run. Trust this value absolutely. Compare every date you see in schema markup, meta tags, article publish dates or last-modified dates against this reference date to decide whether it is past or future. Do NOT rely on your own knowledge of what year "should" be current. If a date is on or before {TODAY_DATE} it is in the past. If it is after {TODAY_DATE} it is in the future.

Score EACH page 1-10 across these 9 dimensions:
1. ARIA – landmark roles, aria-labels, accessibility for AI parsers
2. SCHEMA – schema.org JSON-LD structured data presence and quality.
   The schema summary shows nested objects (address, offers, author, publisher) with a "present" list and, only when relevant, a "missing required" list. Optional sub-fields (e.g. addressRegion, priceValidUntil, author/publisher url, publisher logo) are never listed as missing, they are simply included in "present" when populated and left out entirely when absent. This means a nested object with no "missing required" entry is fully complete, do not describe it as a stub, placeholder or incomplete. Base every completeness judgement strictly on the "missing required" list, and treat anything not flagged there as complete.
   Each SCHEMA_JSON_LD block is a compressed summary, not truncated raw JSON, the header explicitly says so. Do NOT report schema as "truncated" based on the summary format itself.
3. HEADINGS – H1-H6 hierarchy, clarity, topic signal
4. META – Score based on what AI crawlers actually use, not social sharing signals. Use these criteria:
   HIGH-WEIGHT signals (drive most of the score): unique descriptive <title>, meta description, canonical URL, lang attribute, robots directive, viewport. These are what AI crawlers use to understand and cite a page.
   LOW-WEIGHT signals (should contribute at most 1-2 points): Open Graph tags (og:title, og:description, og:image, og:type), Twitter Card tags. These are for social media previews, not AI citation. A page with only OG tags but no proper title or description should score 3-4, not 7-8.
   SCORE 1-3 (Poor): Missing title, no meta description, no canonical, or duplicate meta across many pages.
   SCORE 4-5 (Moderate): Has title and description but they are generic, templated, or missing canonical. OG tags present but core meta weak.
   SCORE 6-7 (Good): Bespoke title and description per page, correct canonical, proper lang, robots configured. OG tags present as a bonus.
   SCORE 8-9 (Excellent): All of the above plus considered use of og:type per content type (e.g. product on PDPs, article on blog), hreflang for international sites, structured meta that reinforces the page topic.
   SCORE 10: Reserved for exemplary implementations across every metadata surface.
   IMPORTANT: Do NOT award high META scores just because Open Graph is comprehensive. OG tags improve social sharing appearance, they do not meaningfully improve AI visibility. The core AI signals are title, description, canonical and lang.
5. LINKS – internal link quality, anchor text, protocol consistency, density
6. ALT TEXT – image alt attribute quality and completeness
7. CRAWL – server-rendered static HTML vs JS dependency
8. LLM – first-hand expertise, named entities, dates, citations, authority signals
9. CONTENT QUALITY – Score this dimension rigorously. Most commercial pages score 3-5, not 7-9. Use these specific criteria:
   SCORE 1-3 (Poor): Content is purely a list of features, specs, or product names with no explanation of why they matter to the buyer. No benefit statements. No answers to "why should I choose this?" or "what problem does this solve?". Thin content that simply labels things (e.g. "Thermostatic shower kit. Chrome finish. 200mm head.").
   SCORE 4-5 (Moderate): Some benefit language present but mostly feature-led. A few "what you get" statements but little "why it matters" context. No use case or buyer scenario addressed. Standard e-commerce copy that describes rather than sells.
   SCORE 6-7 (Good): Clearly benefit-led for most of the page. Addresses "what's in it for me" for the target buyer. Explains why features matter (e.g. not just "thermostatic" but "thermostatic control keeps water at a safe, consistent temperature"). Answers common pre-purchase questions.
   SCORE 8-9 (Excellent): Genuinely user-centric throughout. Addresses the buyer's specific situation, problem or goal. Uses outcome language ("achieve a spa-like experience at home", "save X", "eliminate Y problem"). Anticipates and answers specific buying questions. Cites proof points (reviews, stats, expert recommendation).
   SCORE 10: Reserved for editorial or guide content that is comprehensive, cites sources, names experts, and fully answers a user's question with no gaps.
   IMPORTANT: A product page that simply lists specifications and says "free delivery" scores NO HIGHER than 4. A category page that lists products with no explanatory copy scores 1-2.

   CRITICAL SCORING RULE: Score ONLY on the content you can actually see in the MAIN_CONTENT and BODY_TEXT_SAMPLE fields. Do NOT assume, infer, or hope that expanded content exists beyond what is shown. If the extracted content is thin, the page IS thin — score it accordingly. Never write phrases like "assuming the full description expands on this" or "if the page includes more detail" — if you did not see the content, it does not count. If a product page's visible content is just the product name, price, and a spec list, score 1-3 regardless of what OG or schema description says. Meta and schema descriptions are marketing summaries, they are NOT the on-page content and do not count toward CONTENT QUALITY scoring.

CRITICAL RULES FOR JSON:
- Return ONLY raw JSON. No markdown, no ```json fences, no preamble, no explanation.
- All string values must use double quotes. Never use single quotes inside JSON strings.
- Escape any double quotes inside string values with a backslash.
- Do not include trailing commas after the last item in any array or object.
- Every string value must be on a single line with no literal newlines inside strings.

CRITICAL RULES FOR CONTENT:
- Write in British English throughout (use "optimise" not "optimize", "colour" not "color", "programme" not "program" etc.).
- Never use long dashes (em dashes or en dashes used as sentence breaks). Use a comma, colon, or rewrite the sentence instead.
- Never use the phrase "AI Visibility Practice".
- The executive_summary field MUST be structured with these exact pipe-delimited sections:
  OVERVIEW: 2-3 sentences on overall AI readiness. | STRENGTHS: up to 3 short bullet points of what is working (use * prefix). | GAPS: up to 3 short bullet points of the critical gaps (use * prefix). | VERDICT: 1 sentence on the single most impactful next step.
  Example: "OVERVIEW: Healix.com is built on strong technical foundations. AI crawlers receive full server-rendered HTML on first request. | STRENGTHS: * Bespoke title and meta description on every audited page. * Server-rendered HTML with no JS dependency. * Verifiable first-hand expertise with named clinicians and clients. | GAPS: * Zero schema.org structured data across all audited pages. * Tab and disclosure widgets are invisible to AI parsers. * og:type defaults to website on product pages. | VERDICT: Shipping Organisation, Service and BreadcrumbList schema sitewide is the single biggest unlock available."

Return ONLY valid JSON in exactly this structure:
{
  "company_name": "string",
  "domain": "string",
  "executive_summary": "3-4 paragraph string describing overall AI readiness",
  "average_score": number,
  "dimension_averages": {
    "aria": number, "schema": number, "headings": number, "meta": number,
    "links": number, "alt_text": number, "crawl": number, "llm": number, "content_quality": number
  },
  "pages": [
    {
      "url": "string",
      "title": "string",
      "score": number,
      "verdict": "string (1-2 sentences)",
      "headline_finding": "string (3-5 sentences)",
      "dimensions": {
        "aria": {"score": number, "detail": "string"},
        "schema": {"score": number, "detail": "string"},
        "headings": {"score": number, "detail": "string"},
        "meta": {"score": number, "detail": "string"},
        "links": {"score": number, "detail": "string"},
        "alt_text": {"score": number, "detail": "string"},
        "crawl": {"score": number, "detail": "string"},
        "llm": {"score": number, "detail": "string"},
        "content_quality": {"score": number, "detail": "string"}
      },
      "specific_findings": ["string", "string", "string"]
    }
  ],
  "cross_cutting_themes": [
    {"title": "string", "detail": "string (2-3 paragraphs)"}
  ],
  "recommendations": [
    {
      "priority": "P1|P2|P3",
      "action": "string",
      "impact": "string",
      "effort": "string",
      "owner": "string"
    }
  ],
  "three_quick_wins": [
    {"number": "1", "title": "string", "detail": "string"},
    {"number": "2", "title": "string", "detail": "string"},
    {"number": "3", "title": "string", "detail": "string"}
  ],
  "whats_working": [
    {"point": "string", "detail": "string"}
  ],
  "whats_holding_back": [
    {"point": "string", "detail": "string"},
    {"point": "string", "detail": "string"},
    {"point": "string", "detail": "string"}
  ]
}

IMPORTANT SCORING RULES:
- Each page "score" is the SUM of its 9 dimension scores (each 1-10), so the maximum is 90.
- "average_score" is the average of all page scores (i.e. average of those sums), so it is also out of 90. Do NOT return the average of dimension averages — that would give a number out of 10, which is wrong.
- Example: if one page scores aria:6, schema:2, headings:7, meta:8, links:6, alt_text:4, crawl:8, llm:7, content_quality:5 — its score is 53/90, not 6/10.
- "dimension_averages" are the averages of each dimension across all pages, each still out of 10.

IMPORTANT: The "recommendations" array MUST contain 8-12 items ranked by impact. Every audit has recommendations.
Use priority P1 for highest impact quick wins, P2 for this quarter, P3 for backlog.
Example recommendation format:
{"priority": "P1", "action": "Ship Organization + WebSite JSON-LD sitewide", "impact": "Very high - foundation for all AI citation", "effort": "Low - single template insert", "owner": "Dev"}

The "whats_working" and "whats_holding_back" arrays MUST each contain exactly 3 items.
"""

def clean_json_string(raw: str) -> str:
    """Strip markdown fences and extract the outermost JSON object."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]
    return raw


def repair_and_parse(raw: str) -> dict:
    """Try several strategies to parse potentially malformed JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        import json_repair  # type: ignore
        return json_repair.repair_json(raw, return_objects=True)
    except Exception:
        pass

    # Manual fixes: trailing commas, curly/smart quotes
    fixed = raw
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse Gemini response as JSON. First 300 chars:\n{raw[:300]}")


# Hard cap: max characters of extracted signals we'll send per page.
# ~40k chars per page × 4 pages ≈ 160k chars ≈ 40k tokens — well within
# Gemini 2.5 Flash's 1M input window but keeps generation snappy and
# stops PLPs with runaway product listings from starving other pages.
MAX_SIGNAL_CHARS_PER_PAGE = 40_000


def _cap_signals(signals: str, url: str) -> str:
    """Cap signals to MAX_SIGNAL_CHARS_PER_PAGE with a clear notice."""
    if len(signals) <= MAX_SIGNAL_CHARS_PER_PAGE:
        return signals
    truncated = signals[:MAX_SIGNAL_CHARS_PER_PAGE]
    # Try to end at a section boundary for cleanliness
    last_section = truncated.rfind("\n\n")
    if last_section > MAX_SIGNAL_CHARS_PER_PAGE - 5000:
        truncated = truncated[:last_section]
    original_len = len(signals)
    truncated += (
        f"\n\n[EXTRACTOR_CAP: This page's signal output was truncated at "
        f"{len(truncated)} chars (original was {original_len} chars). This is "
        f"a tool-side cap to keep the audit tractable, NOT a site issue. "
        f"Score dimensions based on what is present in the signals shown above; "
        f"do not report the truncation as a problem with the site itself.]"
    )
    return truncated


def _gemini_safe_generate(model, prompt_text: str, generation_config: dict) -> str:
    """
    Call Gemini and safely extract text.
    Gemini can return responses with no text (safety filter, blocked, empty
    candidates) — accessing response.text on those raises, which used to
    kill the audit silently. This wrapper always returns a string or raises
    with a clear reason.
    """
    try:
        response = model.generate_content(
            prompt_text, generation_config=generation_config
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {type(e).__name__}: {e}")

    # Extract text robustly — response.text may raise if no valid candidate
    try:
        text = response.text
        if text and text.strip():
            return text
    except (ValueError, AttributeError):
        pass

    # Fall back: dig into candidates manually to get a useful error message
    reason_bits = []
    try:
        pf = response.prompt_feedback
        if pf and pf.block_reason:
            reason_bits.append(f"prompt_block_reason={pf.block_reason}")
    except Exception:
        pass
    try:
        for i, c in enumerate(response.candidates or []):
            if hasattr(c, 'finish_reason'):
                reason_bits.append(f"candidate_{i}_finish={c.finish_reason}")
            if hasattr(c, 'safety_ratings') and c.safety_ratings:
                blocked = [s for s in c.safety_ratings if getattr(s, 'blocked', False)]
                if blocked:
                    reason_bits.append(f"candidate_{i}_safety_blocked={len(blocked)}")
            # Try to extract partial text
            if hasattr(c, 'content') and c.content and c.content.parts:
                partial = "".join(getattr(p, 'text', '') for p in c.content.parts)
                if partial.strip():
                    return partial
    except Exception:
        pass

    reason = ", ".join(reason_bits) if reason_bits else "unknown reason (empty response)"
    raise RuntimeError(f"Gemini returned no usable text: {reason}")


def _normalise_audit_shape(result: dict) -> dict:
    """
    Gemini occasionally returns dimensions as a list of objects instead of a
    dict keyed by dimension name, or returns an individual dimension value as
    a list wrapping the real dict. Both shapes cause `.get()` calls
    downstream to raise `AttributeError: 'list' object has no attribute 'get'`.
    Coerce everything back into the expected {dim_key: {score, ...}} shape.
    """
    if not isinstance(result, dict):
        return result
    pages = result.get("pages")
    if not isinstance(pages, list):
        return result
    for page in pages:
        if not isinstance(page, dict):
            continue
        dims = page.get("dimensions")
        # Case 1: dimensions came back as a list of dimension objects
        if isinstance(dims, list):
            fixed = {}
            for item in dims:
                if isinstance(item, dict):
                    key = (item.get("name") or item.get("dimension")
                           or item.get("key") or item.get("id"))
                    if key:
                        fixed[str(key)] = item
            page["dimensions"] = fixed
            dims = fixed
        if not isinstance(dims, dict):
            page["dimensions"] = {}
            continue
        # Case 2: an individual dimension value is a list, not a dict
        for k, v in list(dims.items()):
            if isinstance(v, list):
                dims[k] = v[0] if (v and isinstance(v[0], dict)) else {}
            elif not isinstance(v, dict):
                dims[k] = {}
    # Also normalise dimension_averages if it drifted to a list
    da = result.get("dimension_averages")
    if isinstance(da, list):
        fixed = {}
        for item in da:
            if isinstance(item, dict):
                key = item.get("name") or item.get("dimension") or item.get("key")
                val = item.get("score") or item.get("average") or item.get("value")
                if key is not None and val is not None:
                    fixed[str(key)] = val
        result["dimension_averages"] = fixed
    return result


def _dim_dict(dims, dk):
    """Safe accessor: always returns a dict for a dimension entry."""
    if not isinstance(dims, dict):
        return {}
    v = dims.get(dk)
    return v if isinstance(v, dict) else {}


def run_audit(model, pages: dict) -> dict:
    from datetime import date as _date
    today_iso = _date.today().isoformat()
    today_readable = _date.today().strftime("%d %B %Y")

    # Cap each page's signals to prevent runaway PLP HTML from starving
    # other pages of budget or timing out the Gemini call.
    pages_text = ""
    for url, signals in pages.items():
        capped = _cap_signals(signals, url)
        pages_text += f"\n\n{'='*60}\n{capped}\n"

    # Inject the current date so Gemini has ground truth for past/future checks.
    pages_text += f"\n\n{'='*60}\nTODAY_DATE: {today_iso} ({today_readable})\n"

    prompt = AUDIT_PROMPT.replace("{TODAY_DATE}", today_iso)

    raw_text = _gemini_safe_generate(
        model,
        prompt + "\n\nPages to audit:\n" + pages_text,
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 65536,
        },
    )
    raw = clean_json_string(raw_text)

    try:
        result = repair_and_parse(raw)
    except ValueError:
        # Retry: ask Gemini to fix its own output
        fix_prompt = (
            "The following text should be valid JSON but contains errors. "
            "Return ONLY the corrected JSON object with no other text, "
            "no markdown fences, no explanation:\n\n" + raw[:6000]
        )
        raw_text2 = _gemini_safe_generate(
            model, fix_prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 65536},
        )
        raw2 = clean_json_string(raw_text2)
        result = repair_and_parse(raw2)

    # ── Normalise Gemini's occasional shape drift before any .get() chains ──
    result = _normalise_audit_shape(result)

    # ── Post-process: fill missing recommendations from page findings ──────
    if not result.get("recommendations"):
        recs = []
        dim_keys = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
        dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]
        action_map = {
            "schema":          ("Ship schema.org JSON-LD structured data sitewide",
                                "Very high — biggest single AI citation unlock", "Low — template insert", "Dev"),
            "aria":            ("Add ARIA landmark roles to navigation and content regions",
                                "Medium — improves AI page structure parsing", "Low — template tweak", "Dev"),
            "alt_text":        ("Audit and fix all image alt attributes",
                                "High — dual accessibility and AI win", "Low — CMS field fix", "Content"),
            "headings":        ("Ensure every page has a unique, descriptive H1",
                                "High — primary topic signal for AI crawlers", "Low — template fix", "Dev"),
            "links":           ("Normalise internal links to consistent https:// protocol",
                                "Medium — removes redirect noise for crawlers", "Medium — site-wide pass", "Dev"),
            "meta":            ("Add bespoke meta description to every page",
                                "Medium — strengthens per-page topic signal", "Low — content pass", "Content"),
            "crawl":           ("Audit JS-dependent content and ensure static HTML fallbacks",
                                "High — critical for AI crawler access", "High — architecture review", "Dev"),
            "llm":             ("Add named experts, dates and first-hand detail to key pages",
                                "High — converts pages into citable authority content", "Medium — content pass", "Content"),
            "content_quality": ("Rewrite key pages to lead with user benefits rather than features",
                                "High — benefit-led content is more likely to be cited by AI in answers", "Medium — content rewrite", "Content"),
        }
        # Find lowest-scoring dimensions across all pages
        avg_scores = result.get("dimension_averages", {})
        sorted_dims = sorted(dim_keys, key=lambda k: avg_scores.get(k, 10))
        priority_map = ["P1","P1","P2","P2","P2","P3","P3","P3"]
        for i, dk in enumerate(sorted_dims):
            if dk in action_map:
                action, impact, effort, owner = action_map[dk]
                recs.append({
                    "priority": priority_map[i],
                    "action": action,
                    "impact": impact,
                    "effort": effort,
                    "owner": owner,
                })
        result["recommendations"] = recs

    # Fill whats_working / whats_holding_back if empty
    if not result.get("whats_working"):
        result["whats_working"] = [
            {"point": "Review audit details", "detail": "See per-page dimension breakdown for strengths."}
        ]
    if not result.get("whats_holding_back"):
        dim_keys = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
        avg_scores = result.get("dimension_averages", {})
        worst = sorted(dim_keys, key=lambda k: avg_scores.get(k, 10))[:3]
        labels = {"aria":"ARIA","schema":"Schema","headings":"Headings","meta":"Meta",
                  "links":"Links","alt_text":"Alt Text","crawl":"Crawl","llm":"LLM content"}
        result["whats_holding_back"] = [
            {"point": f"{labels.get(dk,'Unknown')} gap",
             "detail": f"Scoring {avg_scores.get(dk,0)}/10 — a priority improvement area."}
            for dk in worst
        ]

    # ── Always recalculate scores from dimension data (don't trust Gemini's maths) ──
    dim_keys = ["aria", "schema", "headings", "meta", "links", "alt_text", "crawl", "llm", "content_quality"]
    for page in result.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        dims = page.get("dimensions", {})
        dim_scores = [_dim_dict(dims, dk).get("score", 0) or 0 for dk in dim_keys]
        if any((s or 0) > 0 for s in dim_scores):
            page["score"] = sum(dim_scores)   # sum of 8 dims = score out of 80

    # Recalculate dimension averages across all pages
    pages = [p for p in (result.get("pages", []) or []) if isinstance(p, dict)]
    if pages:
        for dk in dim_keys:
            scores = [_dim_dict(p.get("dimensions", {}), dk).get("score", 0) or 0 for p in pages]
            result.setdefault("dimension_averages", {})[dk] = round(sum(scores) / len(scores), 1)
        # Recalculate average_score as average of page totals (out of 80)
        page_totals = [p.get("score", 0) or 0 for p in pages]
        result["average_score"] = round(sum(page_totals) / len(page_totals), 1)

    return result

# ─── Score colour ─────────────────────────────────────────────────────────────
def score_color(s):
    if s <= 2: return "#C0392B"   # red
    if s <= 5: return "#E67E22"   # amber
    return "#27AE60"              # green

# ─── Build Word doc ───────────────────────────────────────────────────────────
def build_docx(data: dict, month_year: str) -> bytes:
    """
    Writes audit data to a JSON file, then runs a static Node.js script
    that reads the JSON. This avoids injecting any user/LLM content into
    JS source code, which was causing SyntaxErrors.
    """
    import tempfile, os, subprocess, json as _json

    # Resolve `node` and `npm` executables.
    # On Streamlit Cloud we install nodejs via the `nodejs-bin` pip package
    # (Debian bullseye's apt is broken since EOL), which bundles the binaries
    # inside its own Python package directory rather than putting them on PATH.
    # Locally, developers may have system Node.js instead. Try the bundled
    # binaries first, then fall back to system PATH.
    def _resolve_node_binaries():
        import shutil, glob
        # 1. Try nodejs-bin — look inside its package dir for the actual binaries.
        #    Layout differs slightly between versions, so we search a few
        #    candidate locations rather than hard-coding one.
        try:
            import nodejs
            pkg_dir = os.path.dirname(os.path.abspath(nodejs.__file__))
            candidate_dirs = [
                pkg_dir,
                os.path.join(pkg_dir, "node_bin"),
                os.path.join(pkg_dir, "bin"),
                os.path.join(pkg_dir, "node", "bin"),
            ]
            for match in glob.glob(os.path.join(pkg_dir, "node-*", "bin")):
                candidate_dirs.append(match)

            # Find node (must be a real, executable binary)
            node_found = None
            for d in candidate_dirs:
                if not os.path.isdir(d):
                    continue
                p = os.path.join(d, "node")
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    node_found = p
                    break

            # Find npm — but PREFER npm-cli.js over the bin/npm wrapper script.
            # The wrapper is a POSIX shell script whose shebang can be broken
            # inside pip-installed layouts ("Exec format error"). npm-cli.js
            # is a plain JavaScript file we can hand straight to node, which
            # is exactly what the wrapper does anyway.
            npm_found = None
            js_candidates = []
            # Common layouts, in preference order
            for base in [pkg_dir] + candidate_dirs:
                js_candidates.append(os.path.join(base, "lib", "node_modules", "npm", "bin", "npm-cli.js"))
                js_candidates.append(os.path.join(base, "node_modules", "npm", "bin", "npm-cli.js"))
                # Relative to a bin/ dir found earlier
                if os.path.basename(base) == "bin":
                    parent = os.path.dirname(base)
                    js_candidates.append(os.path.join(parent, "lib", "node_modules", "npm", "bin", "npm-cli.js"))
            # Last-resort recursive search inside the package
            js_candidates.extend(glob.glob(os.path.join(pkg_dir, "**", "npm-cli.js"), recursive=True))

            for p in js_candidates:
                if os.path.isfile(p):
                    npm_found = p   # returned as a .js path; _run_binary will invoke via node
                    break

            # Absolute fallback: use bin/npm if it happens to actually work
            if not npm_found:
                for d in candidate_dirs:
                    p = os.path.join(d, "npm")
                    if os.path.isfile(p) and os.access(p, os.X_OK):
                        npm_found = p
                        break

            if node_found and npm_found:
                return node_found, npm_found

            # 2. Wrapper API fallback
            if hasattr(nodejs, "node") and hasattr(nodejs, "npm"):
                return ("__NODEJS_BIN_API__", "__NODEJS_BIN_API__")
        except Exception:
            pass

        # 3. System PATH — the "just installed nodejs yourself" case.
        return shutil.which("node") or "node", shutil.which("npm") or "npm"

    NODE_BIN, NPM_BIN = _resolve_node_binaries()

    def _run_binary(kind: str, args: list, **kw):
        """kind is 'node' or 'npm'. Uses nodejs-bin's wrapper API when the
        resolver returned the __NODEJS_BIN_API__ marker; otherwise plain
        subprocess.run against the resolved (or system) binary path."""
        bin_path = NODE_BIN if kind == "node" else NPM_BIN
        if bin_path == "__NODEJS_BIN_API__":
            import nodejs
            wrapper = getattr(nodejs, kind)
            # nodejs-bin exposes .run() that returns a CompletedProcess-like
            # object; older versions only have .call() returning an int.
            if hasattr(wrapper, "run"):
                return wrapper.run(args, **kw)
            code = wrapper.call(args)
            return subprocess.CompletedProcess(
                args=[kind, *args], returncode=code, stdout=b"", stderr=b""
            )
        # For npm-cli.js (a .js file, not an executable) we have to run it
        # through node explicitly.
        if kind == "npm" and bin_path.endswith(".js"):
            node_bin = NODE_BIN if NODE_BIN != "__NODEJS_BIN_API__" else "node"
            return subprocess.run([node_bin, bin_path, *args], **kw)
        return subprocess.run([bin_path, *args], **kw)

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    def score_color_hex(s):
        if s <= 2: return "C0392B"
        if s <= 5: return "E67E22"
        return "27AE60"

    # Enrich data with derived fields the JS needs
    enriched = _json.loads(_json.dumps(data))   # deep copy via json

    # Strip backticks and long dashes from all string values in the data
    def _clean_strings(obj):
        if isinstance(obj, str):
            return obj.replace("`", "").replace("—", ",").replace("–", ",")
        if isinstance(obj, dict):
            return {k: _clean_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_strings(v) for v in obj]
        return obj
    enriched = _clean_strings(enriched)
    enriched["month_year"]   = month_year
    enriched["logo_path"]    = LOGO_PATH
    enriched["dim_keys"]     = dim_keys
    enriched["dim_labels"]   = dim_labels
    # Add colour fields
    # Defensive: dimension_averages / dimensions may occasionally arrive as
    # lists rather than dicts if Gemini shape-drifted. _normalise_audit_shape
    # should have handled this upstream, but re-guard here since build_docx
    # can also be called on cached/loaded audit data.
    dim_avg = enriched.get("dimension_averages", {})
    if not isinstance(dim_avg, dict):
        dim_avg = {}
    enriched["dim_colors"] = {k: score_color_hex(dim_avg.get(k, 0) or 0) for k in dim_keys}
    for page in enriched.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        dims = page.get("dimensions", {})
        if not isinstance(dims, dict):
            dims = {}
            page["dimensions"] = dims
        def _safe_score(dk):
            v = dims.get(dk)
            if isinstance(v, list):
                v = v[0] if (v and isinstance(v[0], dict)) else {}
            if not isinstance(v, dict):
                return 0
            return v.get("score", 0) or 0
        page["dim_colors"] = {k: score_color_hex(_safe_score(k)) for k in dim_keys}
        page["score_color"] = score_color_hex(page.get("score", 0) or 0)

    # Ensure docx npm package is available — install locally if not found globally
    app_dir = os.path.dirname(os.path.abspath(__file__))
    node_modules = os.path.join(app_dir, "node_modules")
    if not os.path.exists(os.path.join(node_modules, "docx")):
        _run_binary("npm",
            ["install", "docx", "--prefix", app_dir],
            capture_output=True, timeout=120
        )

    # Write data JSON to temp file
    data_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    _json.dump(enriched, data_file, ensure_ascii=True)
    data_file.close()

    out_file = "/tmp/audit_output.docx"

    # Static JS template — reads ALL content from the JSON file, zero f-string injection
    js_script = r"""
'use strict';
const fs   = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat
} = require('docx');

const DATA_FILE = process.argv[2];
const OUT_FILE  = process.argv[3];

const d = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));

const company   = d.company_name  || 'Client';
const domain    = d.domain        || '';
const avg       = d.average_score || 0;
const dimAvg    = d.dimension_averages || {};
const pages     = d.pages         || [];
const themes    = d.cross_cutting_themes || [];
const recs      = d.recommendations || [];
const summary   = d.executive_summary || '';
const monthYear = d.month_year    || '';
const logoPath  = d.logo_path     || '';
const dimKeys   = d.dim_keys;
const dimLabels = d.dim_labels;
const dimColors = d.dim_colors;

const logoData = fs.existsSync(logoPath) ? fs.readFileSync(logoPath) : null;

const THIN  = { style: BorderStyle.SINGLE, size: 1,  color: 'DDDDDD' };
const NONE  = { style: BorderStyle.NONE };
const RED_LINE = { style: BorderStyle.SINGLE, size: 4, color: 'D93B1A' };
const borders   = { top: THIN, bottom: THIN, left: THIN, right: THIN };
const noBorders = { top: NONE, bottom: NONE, left: NONE, right: NONE };

function txt(text, opts) {
  return new TextRun(Object.assign({ text: String(text), font: 'Arial', size: 20 }, opts || {}));
}
function para(children, opts) {
  if (!Array.isArray(children)) children = [children];
  return new Paragraph(Object.assign({ children }, opts || {}));
}
function cell(children, width, opts) {
  if (!Array.isArray(children)) children = [children];
  return new TableCell(Object.assign({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children
  }, opts || {}));
}
function hdrCell(label, width, bg) {
  return cell(
    [para([txt(label, { bold: true, color: 'FFFFFF', size: 18 })])],
    width,
    { shading: { fill: bg || 'D93B1A', type: ShadingType.CLEAR } }
  );
}
function scoreCell(score, width) {
  const col = score <= 2 ? 'C0392B' : score <= 5 ? 'E67E22' : '27AE60';
  return cell(
    [para([txt(score + '/10', { bold: true, color: 'FFFFFF', size: 22 })])],
    width,
    { shading: { fill: col, type: ShadingType.CLEAR } }
  );
}
function textCell(text, width, opts) {
  return cell([para([txt(text, { size: 18 })])], width, opts || {});
}
function colorCell(text, width, fill) {
  return cell(
    [para([txt(text, { bold: true, size: 18 })])],
    width,
    { shading: { fill: fill, type: ShadingType.CLEAR } }
  );
}

// ── Header ──────────────────────────────────────────────────────────────────
const NO_TABLE_BORDER = {
  top: NONE, bottom: NONE, left: NONE, right: NONE,
  insideH: NONE, insideV: NONE,
};
const headerChildren = [
  new Table({
    width: { size: 9906, type: WidthType.DXA },
    columnWidths: [1200, 8706],
    borders: NO_TABLE_BORDER,
    rows: [new TableRow({ children: [
      new TableCell({ borders: noBorders, width: { size: 1200, type: WidthType.DXA },
        children: logoData ? [para([new ImageRun({ data: logoData, transformation: { width: 50, height: 50 }, type: 'png' })])] : [para([txt('')])] }),
      new TableCell({ borders: noBorders, width: { size: 8706, type: WidthType.DXA },
        verticalAlign: VerticalAlign.CENTER,
        children: [para([txt('AI VISIBILITY AUDIT \u00b7 ' + company.toUpperCase() + ' \u00b7 ' + monthYear.toUpperCase(), { size: 16, color: '6B6B6B' })], { alignment: AlignmentType.RIGHT })] }),
    ]})]
  }),
  para([txt('')], { border: { bottom: RED_LINE } }),
];

// ── Footer ──────────────────────────────────────────────────────────────────
const footerChildren = [
  para([txt('')], { border: { top: RED_LINE } }),
  para([
    txt('SUMMITMEDIA.CO.UK', { size: 16, color: '6B6B6B' }),
    txt('\t\tSUMMITMEDIA.CO.UK', { size: 16, color: '6B6B6B' }),
    txt('\t\t', { size: 16 }),
    new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 16, color: '6B6B6B' }),
  ]),
];

// ── Scorecard table ──────────────────────────────────────────────────────────
const scorecardRows = [
  new TableRow({ tableHeader: true, children: [
    hdrCell('PAGE', 3500), hdrCell('SCORE', 900), hdrCell('VERDICT', 4960)
  ]})
];
pages.forEach(function(p, i) {
  const fill = i % 2 === 0 ? 'F5F4F2' : 'FFFFFF';
  const shade = { shading: { fill, type: ShadingType.CLEAR } };
  scorecardRows.push(new TableRow({ children: [
    cell([para([txt(p.title || ('Page '+(i+1)), { bold: true, size: 18 })])], 3500, shade),
    cell([para([txt((p.score||0)+'/90', { bold: true, size: 18 })], { alignment: AlignmentType.CENTER })], 900, shade),
    cell([para([txt(p.verdict || '', { size: 18 })])], 4960, shade),
  ]}));
});

// ── Dimension averages row ───────────────────────────────────────────────────
const dimAvgCells = dimKeys.map(function(dk, i) {
  const s = dimAvg[dk] || 0;
  const col = dimColors[dk] || '27AE60';
  return cell([
    para([txt(dimLabels[i], { size: 13, bold: true, color: 'FFFFFF' })], { alignment: AlignmentType.CENTER }),
    para([txt(s+'/10', { size: 22, bold: true, color: 'FFFFFF' })], { alignment: AlignmentType.CENTER }),
  ], 1040, { shading: { fill: col, type: ShadingType.CLEAR } });
});

// ── Per-page sections ────────────────────────────────────────────────────────
const pageSections = [];
pages.forEach(function(p, i) {
  const dims = p.dimensions || {};
  const dimRows = dimKeys.map(function(dk, di) {
    const dim = dims[dk] || {};
    const s   = dim.score || 0;
    const col = (p.dim_colors || {})[dk] || '27AE60';
    return new TableRow({ children: [
      cell([para([txt(dimLabels[di], { bold: true, color: 'FFFFFF', size: 18 })])], 1200,
        { shading: { fill: col, type: ShadingType.CLEAR } }),
      cell([para([txt(s+'/10', { bold: true, size: 20 })], { alignment: AlignmentType.CENTER })], 600, {}),
      cell([para([txt(dim.detail || '', { size: 18 })])], 7160, {}),
    ]});
  });

  const sfParas = (p.specific_findings || []).map(function(sf) {
    return para([txt(sf, { size: 20 })], { numbering: { reference: 'bullets', level: 0 } });
  });
  if (!sfParas.length) sfParas.push(para([txt('No specific findings recorded.', { size: 20 })]));

  pageSections.push(
    para([txt('PAGE '+(i+1)+': '+(p.title||'').toUpperCase(), { size: 28, bold: true, color: 'D93B1A' })],
      { heading: HeadingLevel.HEADING_1 }),
    para([txt('URL: ', { bold: true, size: 20 }), txt(p.url||'', { size: 20, color: 'D93B1A' })]),
    para([txt('Total score: '+(p.score||0)+'/90', { bold: true, size: 20 })]),
    para([txt(p.headline_finding || '', { size: 20, italics: true })], {
      spacing: { before: 160 },
      border: { left: { style: BorderStyle.SINGLE, size: 20, color: 'D93B1A' } },
      indent: { left: 360 }
    }),
    para([txt('Dimension Breakdown', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
    new Table({ width: { size: 8960, type: WidthType.DXA }, columnWidths: [1200, 600, 7160], rows: dimRows }),
    para([txt('Specific Findings', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
    ...sfParas,
    para([new PageBreak()])
  );
});

// ── Themes ───────────────────────────────────────────────────────────────────
const themeParas = [];
themes.forEach(function(t) {
  themeParas.push(
    para([txt(t.title||'', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
    para([txt(t.detail||'', { size: 20 })], { spacing: { before: 80, after: 160 } })
  );
});
if (!themeParas.length) themeParas.push(para([txt('No cross-cutting themes identified.', { size: 20 })]));

// ── Recommendations table ────────────────────────────────────────────────────
const recRows = [
  new TableRow({ tableHeader: true, children: [
    hdrCell('PRI.', 600), hdrCell('ACTION', 4200), hdrCell('IMPACT', 1800),
    hdrCell('EFFORT', 1500), hdrCell('OWNER', 1260)
  ]})
];
recs.forEach(function(r, i) {
  const fill = i % 2 === 0 ? 'F5F4F2' : 'FFFFFF';
  const shade = { shading: { fill, type: ShadingType.CLEAR } };
  recRows.push(new TableRow({ children: [
    cell([para([txt(r.priority||'', { bold: true, size: 18, color: 'D93B1A' })])], 600, shade),
    cell([para([txt(r.action||'', { size: 18 })])], 4200, shade),
    cell([para([txt(r.impact||'', { size: 18 })])], 1800, shade),
    cell([para([txt(r.effort||'', { size: 18 })])], 1500, shade),
    cell([para([txt(r.owner||'', { size: 18 })])], 1260, shade),
  ]}));
});

// ── Executive Summary renderer ───────────────────────────────────────────────
function buildExecutiveSummary(raw) {
  var paras = [];
  // Parse pipe-delimited sections: OVERVIEW: ... | STRENGTHS: ... | GAPS: ... | VERDICT: ...
  var sections = raw.split('|').map(function(s) { return s.trim(); });
  sections.forEach(function(section) {
    var colonIdx = section.indexOf(':');
    if (colonIdx === -1) {
      // No section label — just output as normal paragraph
      if (section.trim()) {
        paras.push(para([txt(section.trim(), { size: 20 })], { spacing: { before: 80, after: 100 } }));
      }
      return;
    }
    var label = section.substring(0, colonIdx).trim().toUpperCase();
    var body  = section.substring(colonIdx + 1).trim();

    // Section heading
    paras.push(para([txt(label, { size: 20, bold: true, color: 'D93B1A' })],
      { spacing: { before: 200, after: 60 } }));

    // Split body into bullet lines (* prefix) and normal lines
    var lines = body.split('*').map(function(l) { return l.trim(); }).filter(Boolean);
    if (lines.length > 1) {
      // Multiple items = bullet list
      lines.forEach(function(line) {
        paras.push(para([txt(line, { size: 20 })],
          { numbering: { reference: 'bullets', level: 0 }, spacing: { after: 40 } }));
      });
    } else {
      // Single block = normal paragraph
      paras.push(para([txt(body, { size: 20 })],
        { spacing: { before: 40, after: 100 } }));
    }
  });

  // Fallback: if parsing produced nothing useful, just show raw text
  if (paras.length === 0) {
    paras.push(para([txt(raw, { size: 20 })], { spacing: { before: 80, after: 160 } }));
  }
  return paras;
}

// ── Document ─────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial', color: '1A1A1A' },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Arial', color: '1A1A1A' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1000, right: 1000, bottom: 1000, left: 1000 }
      }
    },
    headers: { default: new Header({ children: headerChildren }) },
    footers: { default: new Footer({ children: footerChildren }) },
    children: [
      // Cover
      para([txt('AI VISIBILITY AUDIT', { size: 56, bold: true })], { spacing: { before: 400 } }),
      para([txt(company, { size: 40, bold: true, color: 'D93B1A' })]),
      para([txt('Technical readiness for the AI search era', { size: 24, italics: true, color: '6B6B6B' })],
        { spacing: { after: 320 } }),
      new Table({
        width: { size: 4000, type: WidthType.DXA },
        columnWidths: [1800, 2200],
        rows: [
          new TableRow({ children: [
            cell([para([txt('Domain', { bold: true, size: 18 })])], 1800,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
            cell([para([txt(domain, { size: 18, color: 'D93B1A' })])], 2200,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
          ]}),
          new TableRow({ children: [
            cell([para([txt('Audit date', { bold: true, size: 18 })])], 1800,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
            cell([para([txt(monthYear, { size: 18 })])], 2200,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
          ]}),
          new TableRow({ children: [
            cell([para([txt('Prepared by', { bold: true, size: 18 })])], 1800,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
            cell([para([txt('Summit Media', { size: 18 })])], 2200,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
          ]}),
        ]
      }),
      para([txt('Confidential. Prepared for the ' + company + ' digital team.', { size: 18, italics: true, color: '6B6B6B' })],
        { spacing: { before: 160 } }),
      para([new PageBreak()]),

      // Executive Summary — parse the structured pipe-delimited format
      para([txt('EXECUTIVE SUMMARY', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      ...buildExecutiveSummary(summary),

      // Scorecard
      para([txt('Headline Scorecard', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3500, 900, 4960], rows: scorecardRows }),

      // Dimension averages
      para([txt('Dimension Averages', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2, spacing: { before: 320 } }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040],
        rows: [new TableRow({ children: dimAvgCells })]
      }),
      para([new PageBreak()]),

      // Methodology
      para([txt('METHODOLOGY', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      para([txt('This audit treats ' + domain + '\'s site the way ChatGPT, Perplexity, Gemini and Claude actually consume it. AI crawlers fetch a URL, parse the static HTML on first request, look for structured data, and decide whether the page is citable. They do not execute a full JavaScript render in most cases.', { size: 20 })]),
      para([txt('The Eight Dimensions', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      ...['ARIA implementation. Semantic landmarks, role attributes, descriptive aria-label values.',
          'Structured data / schema markup. Schema.org JSON-LD — the single highest-leverage signal for AI citation.',
          'Heading structure. A clean H1\u2013H6 outline that gives crawlers a topical map of the page.',
          'Meta and SEO signals. Title, description, canonical, Open Graph, Twitter Card, robots, language.',
          'Link quality. Internal consistency, descriptive anchor text, protocol uniformity, link density.',
          'Image alt text. Descriptive alt attributes \u2014 the cheapest accessibility-and-AI dual win available.',
          'Crawlability and JS dependency. Whether content is present in static HTML or requires JS execution.',
          'LLM content signals. First-hand expertise, named authors, dates, citations, accreditations.',
          'Content quality. Benefit-led writing, clear value propositions, user-focused language, and direct answers to likely user questions.',
      ].map(function(t) { return para([txt(t, { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }); }),
      para([txt('Scoring Bands', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      para([txt('Red (1\u20132): Critical. Actively blocking AI visibility. Fix first.', { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }),
      para([txt('Amber (3\u20135): Capping. Under-performing relative to what\u2019s possible. High value to fix.', { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }),
      para([txt('Green (6+): Working. Meets the bar AI crawlers expect.', { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }),
      para([new PageBreak()]),

      // Per-page sections
      ...pageSections,

      // Cross-cutting themes
      para([txt('CROSS-CUTTING THEMES', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      ...themeParas,
      para([new PageBreak()]),

      // Recommendations
      para([txt('PRIORITY RECOMMENDATIONS', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      para([txt('Ranked by AI-citation impact relative to implementation cost. P1 = ship in the next sprint. P2 = ship this quarter. P3 = ship when convenient.', { size: 20, italics: true })],
        { spacing: { after: 160 } }),
      new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 4200, 1800, 1500, 1260], rows: recRows }),
    ]
  }]
});

Packer.toBuffer(doc).then(function(buf) {
  fs.writeFileSync(OUT_FILE, buf);
  console.log('OK');
}).catch(function(e) { console.error(e); process.exit(1); });
"""

    script_file = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
    script_file.write(js_script)
    script_file.close()

    out_path = "/tmp/audit_output.docx"
    app_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    # Include both local (app dir) and global npm paths
    local_modules  = os.path.join(app_dir, "node_modules")
    global_modules = _run_binary("npm",
        ["root", "-g"], capture_output=True, text=True
    ).stdout.strip()
    env["NODE_PATH"] = local_modules + os.pathsep + global_modules
    result = _run_binary("node",
        [script_file.name, data_file.name, out_path],
        capture_output=True, text=True, timeout=90,
        env=env
    )
    os.unlink(script_file.name)
    os.unlink(data_file.name)

    if result.returncode != 0:
        raise RuntimeError(f"docx generation failed:\n{result.stderr[-2000:]}")

    with open(out_path, "rb") as f:
        return f.read()




# ─── Build One-Pager PDF ──────────────────────────────────────────────────────
def build_onepager(data: dict, month_year: str) -> bytes:
    """
    Guaranteed single-page A4 PDF.
    All sections use fixed rowHeights so text is clipped, never expanding the page.
    """
    company      = data.get("company_name", "Client")
    domain       = data.get("domain", "")
    avg          = round(data.get("average_score", 0))
    dim_avg      = data.get("dimension_averages", {})
    exec_summary = data.get("executive_summary", "")
    working      = data.get("whats_working", [])[:3]
    holding      = data.get("whats_holding_back", [])[:3]
    wins         = data.get("three_quick_wins", [])[:3]

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    RED   = colors.HexColor("#D93B1A")
    DARK  = colors.HexColor("#1A1A1A")
    GREY  = colors.HexColor("#6B6B6B")
    LIGHT = colors.HexColor("#F5F4F2")
    GREEN = colors.HexColor("#27AE60")
    WHITE = colors.white

    def dim_color(s):
        if s <= 2: return colors.HexColor("#C0392B")
        if s <= 5: return colors.HexColor("#E67E22")
        return colors.HexColor("#27AE60")

    def safe(t):
        return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def clean(t):
        t = str(t).replace("`","").replace("\u2014",",").replace("\u2013",",").replace("  "," ")
        return safe(t.strip())

    # ── Page geometry ────────────────────────────────────────────────
    PAGE_W, PAGE_H = A4          # 595 x 842 pt
    ML = MR = 13*mm
    MT = MB = 10*mm
    W  = PAGE_W - ML - MR        # usable width
    H  = PAGE_H - MT - MB        # usable height ~822pt

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
    )

    # ── Style factory ────────────────────────────────────────────────
    _sc = {}
    def S(name, **kw):
        key = name + str(sorted(kw.items()))
        if key not in _sc:
            base = dict(fontName="Helvetica", fontSize=8, leading=10,
                        textColor=DARK, spaceAfter=0, spaceBefore=0)
            base.update(kw)
            _sc[key] = ParagraphStyle(name + str(len(_sc)), **base)
        return _sc[key]

    def P(markup, **kw):
        return Paragraph(markup, S("p", **kw))

    # ── Fixed height budget (pts) ───────────────────────────────────
    # Coded values are scaled so total ~= usable page height (785pt).
    # ReportLab adds internal padding on top, so these values are tuned
    # so the content fills the page without overflowing.
    #
    # Critical-bot callout: only rendered when a bot in CRITICAL_AI_BOTS
    # has been blocked. When shown, we shave a little off the score box,
    # WH content rows and wins row so total budget is unchanged.
    _blocked_critical_pdf = get_blocked_critical_bots(data.get("bot_access", {}))
    _show_alert = bool(_blocked_critical_pdf)

    ROW_ALERT = 22 if _show_alert else 0    # red banner right below header
    ROW_HDR  = 33    # logo header
    ROW_HERO = 84    # headline + intro paragraph
    ROW_SCOR = 78 if _show_alert else 84    # score box
    ROW_DIMS = 28    # dimension badge strip
    ROW_WHHD = 29    # WH section header row
    ROW_WH   = 89 if _show_alert else 94    # each WH content row (x3)
    ROW_WINS = 25    # "quick wins" label row
    ROW_WNUM = 146   # wins content row — number + title + detail
    ROW_CTA  = 35    # CTA bar
    ROW_FOOT = 13    # footer

    # Spacers (pts)
    SP1 = 5   # after header
    SP2 = 4   # after hero
    SP3 = 4   # after score box
    SP4 = 4   # after dims
    SP5 = 4   # after WH section
    SP6 = 3   # after wins

    # ═══════════════════════════════════════════════════════════════
    # HELPER: Table cell that clips content to a fixed height
    # ReportLab clips automatically when rowHeights is specified and
    # splitByRow=0 is not set — we just need to specify the heights.
    # ═══════════════════════════════════════════════════════════════

    story = []

    # ═══════════════════════════════════════════════════════════════
    # 1. HEADER
    # ═══════════════════════════════════════════════════════════════
    LOGO_SZ = 26
    logo_cell = RLImage(LOGO_PATH, width=LOGO_SZ, height=LOGO_SZ) if os.path.exists(LOGO_PATH) else P("")
    hdr_t = Table(
        [[logo_cell,
          P(f'<font name="Helvetica" size="6" color="#6B6B6B">AI VISIBILITY SNAPSHOT<br/>'
            f'{safe(company).upper()} &middot; {safe(month_year).upper()}</font>',
            alignment=TA_RIGHT, leading=8)]],
        colWidths=[LOGO_SZ + 3*mm, W - LOGO_SZ - 3*mm],
        rowHeights=[ROW_HDR - 4],
    )
    hdr_t.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),2),
        ("LINEBELOW",     (0,0),(-1,-1),1.5,RED),
    ]))
    story.append(hdr_t)
    story.append(Spacer(1, SP1))

    # ═══════════════════════════════════════════════════════════════
    # 1b. CRITICAL BOT ALERT (only if any critical AI bot is blocked)
    # ═══════════════════════════════════════════════════════════════
    if _show_alert:
        # Build short product list ("ChatGPT, Claude and Perplexity")
        _products = sorted({b["product"] for b in _blocked_critical_pdf})
        if len(_products) > 1:
            _prod_str = ", ".join(_products[:-1]) + " and " + _products[-1]
        else:
            _prod_str = _products[0]
        _uas = ", ".join(b["user_agent"] for b in _blocked_critical_pdf)
        alert_t = Table(
            [[P(
                f'<font name="Helvetica-Bold" size="8" color="#FFFFFF">! CRITICAL — '
                f'{safe(_prod_str)} cannot crawl this site.</font> '
                f'<font name="Helvetica" size="7" color="#FFFFFF">'
                f'Blocked: {safe(_uas)}. AI answers won\u2019t cite pages the crawlers can\u2019t fetch.</font>',
                leading=10,
            )]],
            colWidths=[W],
            rowHeights=[ROW_ALERT],
        )
        alert_t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), RED),
            ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ]))
        story.append(alert_t)

    # ═══════════════════════════════════════════════════════════════
    # 2. HERO HEADLINE + INTRO  (fixed height table so it cannot grow)
    # ═══════════════════════════════════════════════════════════════
    if 'OVERVIEW:' in exec_summary.upper():
        ov = exec_summary.split('|')[0]
        ci = ov.find(':')
        intro_raw = clean(ov[ci+1:].strip()) if ci != -1 else clean(ov)
    else:
        intro_raw = clean(exec_summary)
    # 165 chars fits ~2 lines at 7pt/9lead within ROW_HERO
    intro_txt = intro_raw  # no cap — hero row height clips naturally

    hero_t = Table(
        [[P('<font name="Helvetica-Bold" size="17">Is your site ready '
            f'for the <font color="#D93B1A"><i>AI search era?</i></font></font>',
            leading=20)],
         [P(f'We audited <b>{safe(domain)}</b> the way ChatGPT, Perplexity, Gemini and Claude see it. '
            + intro_txt, fontSize=7, leading=9)]],
        colWidths=[W],
        rowHeights=[22, ROW_HERO - 22],
    )
    hero_t.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),2),
    ]))
    story.append(hero_t)
    story.append(Spacer(1, SP2))

    # ═══════════════════════════════════════════════════════════════
    # 3. SCORE BOX  — fixed height, verdict clipped to 1 line
    # ═══════════════════════════════════════════════════════════════
    verdict_text = ""
    if "VERDICT:" in exec_summary.upper():
        for part in exec_summary.split("|"):
            if "VERDICT:" in part.upper():
                ci = part.find(":")
                verdict_text = clean(part[ci+1:].strip()) if ci != -1 else ""
                break

    label_html = (
        '<font name="Helvetica" size="6.5" color="#6B6B6B">AVERAGE PAGE SCORE</font><br/>'
        '<font name="Helvetica-Bold" size="12" color="#1A1A1A">A solid foundation. </font>'
        '<font name="Helvetica-Bold" size="12" color="#D93B1A">A clear AI gap.</font>'
    )
    if verdict_text:
        label_html += f'<br/><font name="Helvetica" size="6.5" color="#6B6B6B">{verdict_text}</font>'

    score_t = Table(
        [[P(f'<font name="Helvetica-Bold" size="36" color="#D93B1A">{avg}</font>'
            f'<font name="Helvetica" size="12" color="#6B6B6B">/90</font>',
            alignment=TA_CENTER, leading=40),
          P(label_html, leading=13)]],
        colWidths=[34*mm, W - 34*mm],
        rowHeights=[ROW_SCOR],
    )
    score_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1),LIGHT),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1),4),
        ("BOTTOMPADDING", (0,0),(-1,-1),4),
        ("LEFTPADDING",   (0,0),(0,0),  6),
        ("LEFTPADDING",   (1,0),(1,0),  8),
        ("RIGHTPADDING",  (0,0),(-1,-1),6),
    ]))
    story.append(score_t)
    story.append(Spacer(1, SP3))

    # ═══════════════════════════════════════════════════════════════
    # 4. DIMENSION BADGES  — fixed height
    # ═══════════════════════════════════════════════════════════════
    cw = W / 9
    dim_cells = []
    for dk, dl in zip(dim_keys, dim_labels):
        s = dim_avg.get(dk, 0)
        dim_cells.append(P(
            f'<font name="Helvetica" size="4.5" color="#FFFFFF">{dl}<br/></font>'
            f'<font name="Helvetica-Bold" size="10" color="#FFFFFF">{s}</font>'
            f'<font name="Helvetica" size="5.5" color="#FFFFFF">/10</font>',
            alignment=TA_CENTER, leading=9,
        ))
    dim_t = Table([dim_cells], colWidths=[cw]*9, rowHeights=[ROW_DIMS])
    ds = [("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
          ("LEFTPADDING",(0,0),(-1,-1),1),("RIGHTPADDING",(0,0),(-1,-1),1)]
    for i, dk in enumerate(dim_keys):
        ds.append(("BACKGROUND",(i,0),(i,0),dim_color(dim_avg.get(dk,0))))
    dim_t.setStyle(TableStyle(ds))
    story.append(dim_t)
    story.append(Spacer(1, SP4))

    # ═══════════════════════════════════════════════════════════════
    # 5. WHAT'S WORKING / HOLDING BACK
    #    Fixed rowHeights on every row — content is clipped, never grows
    # ═══════════════════════════════════════════════════════════════
    GAP  = 4*mm
    HALF = (W - GAP) / 2
    # detail text char limit: ~100 chars fits 2 lines at 6.5pt within ROW_WH
    DET_CAP = 999  # no cap — row height clips naturally

    def build_side(items, hdr_text, hdr_color, icon, icon_color_hex):
        row_heights = [ROW_WHHD]
        rows = [[P(f'<font name="Helvetica-Bold" size="8" color="#FFFFFF">{hdr_text}</font>',
                   leading=10)]]
        for it in (items + [{}, {}, {}])[:3]:
            pt  = clean(it.get("point","")) if it else ""
            det = clean(it.get("detail",""))[:DET_CAP] if it else ""
            content = (
                f'<font name="Helvetica-Bold" size="7.5" color="{icon_color_hex}">{icon} </font>'
                f'<font name="Helvetica-Bold" size="7.5">{pt}</font>'
                + (f'<br/><font name="Helvetica" size="6.5" color="#4A4A4A">{det}</font>' if det else "")
            ) if pt else " "
            rows.append([P(content, leading=9)])
            row_heights.append(ROW_WH)

        t = Table(rows, colWidths=[HALF], rowHeights=row_heights)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  hdr_color),
            ("BACKGROUND",    (0,1),(-1,-1), colors.HexColor("#F9F9F9")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("LINEBELOW",     (0,1),(-1,-2), 0.4, colors.HexColor("#E0E0E0")),
        ]))
        return t

    working_t = build_side(working, "\u271a WHAT\u2019S WORKING",     GREEN, "\u271a", "#27AE60")
    holding_t = build_side(holding, "! WHAT\u2019S HOLDING YOU BACK", RED,   "!",      "#D93B1A")

    sides = Table([[working_t, Spacer(GAP,1), holding_t]],
                  colWidths=[HALF, GAP, HALF])
    sides.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),0),
    ]))
    story.append(sides)
    story.append(Spacer(1, SP5))

    # ═══════════════════════════════════════════════════════════════
    # 6. THREE QUICK WINS  — fixed height per win cell
    # ═══════════════════════════════════════════════════════════════
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#DDDDDD"), spaceAfter=2))
    # Section label + heading in one fixed-height row
    label_t = Table(
        [[P('<font name="Helvetica" size="5.5" color="#6B6B6B">THREE MOVES THAT MOVE THE NEEDLE</font><br/>'
            '<font name="Helvetica-Bold" size="11">Quick wins, big impact</font>',
            leading=11)]],
        colWidths=[W], rowHeights=[ROW_WINS],
    )
    label_t.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),2),
        ("BOTTOMPADDING", (0,0),(-1,-1),2),
    ]))
    story.append(label_t)

    THIRD = W / 3
    # detail cap: ~115 chars fits ~3 lines at 7pt/9lead within ROW_WNUM
    WIN_DET_CAP = 999  # no cap — row height clips naturally
    win_cells = []
    for w in (wins + [{},{},{}])[:3]:
        num    = clean(w.get("number","")) if w else ""
        title  = clean(w.get("title",""))  if w else ""
        detail = clean(w.get("detail",""))[:WIN_DET_CAP] if w else ""
        win_cells.append(P(
            f'<font name="Helvetica-Bold" size="20" color="#D93B1A">{num}</font><br/>'
            f'<font name="Helvetica-Bold" size="7.5">{title}</font><br/>'
            f'<font name="Helvetica" size="7" color="#4A4A4A">{detail}</font>',
            leading=10,
        ) if num else P(" "))

    wins_t = Table([win_cells], colWidths=[THIRD]*3, rowHeights=[ROW_WNUM])
    wins_t.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"TOP"),
        ("TOPPADDING",    (0,0),(-1,-1),2),
        ("BOTTOMPADDING", (0,0),(-1,-1),0),
        ("LEFTPADDING",   (0,0),(0,0),  0),
        ("LEFTPADDING",   (1,0),(-1,-1),8),
        ("RIGHTPADDING",  (0,0),(-1,-1),6),
        ("LINEAFTER",     (0,0),(1,-1), 0.5, colors.HexColor("#DDDDDD")),
    ]))
    story.append(wins_t)
    story.append(Spacer(1, SP6))

    # ═══════════════════════════════════════════════════════════════
    # 7. CTA BAR  — fixed height
    # ═══════════════════════════════════════════════════════════════
    cta_t = Table(
        [[P('<font name="Helvetica-Bold" size="9.5">We\u2019ll walk your team through<br/>'
            'every finding. <font color="#D93B1A"><i>No obligation.</i></font></font>',
            leading=13),
          P('<font name="Helvetica" size="6" color="#6B6B6B">BOOK A SESSION<br/></font>'
            '<font name="Helvetica-Bold" size="10">hello@summitmedia.com</font>',
            alignment=TA_CENTER, leading=12)]],
        colWidths=[W*0.52, W*0.48],
        rowHeights=[ROW_CTA],
    )
    cta_t.setStyle(TableStyle([
        ("BACKGROUND",    (1,0),(1,0),  LIGHT),
        ("TOPPADDING",    (0,0),(-1,-1),6),
        ("BOTTOMPADDING", (0,0),(-1,-1),6),
        ("LEFTPADDING",   (0,0),(0,0),  10),
        ("LEFTPADDING",   (1,0),(1,0),  8),
        ("RIGHTPADDING",  (0,0),(-1,-1),8),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("LINEABOVE",     (0,0),(-1,0), 2, RED),
    ]))
    story.append(cta_t)

    # Footer removed — CTA bar is the final element

    doc.build(story)
    return buf.getvalue()


# ─── Build One-Pager Word doc (editable) ──────────────────────────────────────
def build_onepager_docx(data: dict, month_year: str) -> bytes:
    """
    Editable Word version of the one-pager PDF. Keeps the same seven sections
    (header, hero, score box, dimension badges, what's working / holding back,
    quick wins, CTA) plus a short AI-bot-access summary, but as native Word
    objects so the team can tweak copy before sending it out.
    """
    company      = data.get("company_name", "Client")
    domain       = data.get("domain", "")
    avg          = round(data.get("average_score", 0))
    dim_avg      = data.get("dimension_averages", {}) or {}
    if not isinstance(dim_avg, dict):
        dim_avg = {}
    exec_summary = data.get("executive_summary", "") or ""
    working      = (data.get("whats_working", []) or [])[:3]
    holding      = (data.get("whats_holding_back", []) or [])[:3]
    wins         = (data.get("three_quick_wins", []) or [])[:3]
    bot_access   = data.get("bot_access", {}) or {}

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    RED   = RGBColor(0xD9, 0x3B, 0x1A)
    DARK  = RGBColor(0x1A, 0x1A, 0x1A)
    GREY  = RGBColor(0x6B, 0x6B, 0x6B)
    GREEN = RGBColor(0x27, 0xAE, 0x60)
    AMBER = RGBColor(0xE6, 0x7E, 0x22)
    RED2  = RGBColor(0xC0, 0x39, 0x2B)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    LIGHT_HEX = "F5F4F2"
    GREY_HEX  = "DDDDDD"

    def dim_hex(s):
        try:
            s = float(s)
        except (TypeError, ValueError):
            s = 0
        if s <= 2: return "C0392B"
        if s <= 5: return "E67E22"
        return "27AE60"

    def clean(t):
        return str(t or "").replace("`", "").replace("\u2014", ",").replace("\u2013", ",").strip()

    # ── Document setup ──────────────────────────────────────────────
    doc = Document()

    # Page margins (approx match to PDF: 13mm sides, 10mm top/bottom)
    for section in doc.sections:
        section.top_margin    = Mm(12)
        section.bottom_margin = Mm(12)
        section.left_margin   = Mm(14)
        section.right_margin  = Mm(14)

    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    def _shade(cell, hex_color):
        """Add background shading to a table cell."""
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tc_pr.append(shd)

    def _remove_cell_margins(cell, top=60, bottom=60, left=100, right=100):
        """Set cell margins in twentieths of a point."""
        tc_pr = cell._tc.get_or_add_tcPr()
        mar = OxmlElement("w:tcMar")
        for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
            node = OxmlElement(f"w:{side}")
            node.set(qn("w:w"), str(val))
            node.set(qn("w:type"), "dxa")
            mar.append(node)
        tc_pr.append(mar)

    def _cell_border(cell, side, size_eighths=8, color="D93B1A"):
        """Add a single border to a cell (single line, size in 1/8 pt)."""
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = tc_pr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tc_pr.append(borders)
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), str(size_eighths))
        border.set(qn("w:color"), color)
        borders.append(border)

    def _add_run(paragraph, text, *, bold=False, italic=False, size=None, color=None, font=None):
        run = paragraph.add_run(text)
        run.bold = bold
        run.italic = italic
        if size is not None:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
        if font is not None:
            run.font.name = font
        return run

    def _clear_paragraph_spacing(p):
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after  = Pt(0)

    # ═══════════════════════════════════════════════════════════════
    # 1. HEADER — logo left, snapshot label right, red rule underneath
    # ═══════════════════════════════════════════════════════════════
    hdr_tbl = doc.add_table(rows=1, cols=2)
    hdr_tbl.autofit = False
    hdr_tbl.columns[0].width = Cm(3)
    hdr_tbl.columns[1].width = Cm(15.5)

    logo_cell = hdr_tbl.cell(0, 0)
    logo_p = logo_cell.paragraphs[0]
    _clear_paragraph_spacing(logo_p)
    if os.path.exists(LOGO_PATH):
        try:
            logo_p.add_run().add_picture(LOGO_PATH, width=Cm(1.6))
        except Exception:
            _add_run(logo_p, "SUMMIT", bold=True, size=11, color=RED)
    else:
        _add_run(logo_p, "SUMMIT", bold=True, size=11, color=RED)

    label_cell = hdr_tbl.cell(0, 1)
    label_p = label_cell.paragraphs[0]
    label_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _clear_paragraph_spacing(label_p)
    _add_run(label_p, "AI VISIBILITY SNAPSHOT\n", size=7, color=GREY)
    _add_run(label_p, f"{company.upper()} · {month_year.upper()}", size=7, color=GREY)
    _cell_border(label_cell, "bottom", size_eighths=12, color="D93B1A")
    _cell_border(logo_cell,  "bottom", size_eighths=12, color="D93B1A")

    doc.add_paragraph()  # spacer

    # ═══════════════════════════════════════════════════════════════
    # 1b. CRITICAL BOT ALERT (only if any critical AI bot is blocked)
    # ═══════════════════════════════════════════════════════════════
    _blocked_critical_docx = get_blocked_critical_bots(bot_access)
    if _blocked_critical_docx:
        _products = sorted({b["product"] for b in _blocked_critical_docx})
        if len(_products) > 1:
            _prod_str = ", ".join(_products[:-1]) + " and " + _products[-1]
        else:
            _prod_str = _products[0]
        _uas = ", ".join(b["user_agent"] for b in _blocked_critical_docx)

        alert_tbl = doc.add_table(rows=1, cols=1)
        alert_tbl.autofit = False
        alert_tbl.columns[0].width = Cm(18.5)
        acell = alert_tbl.cell(0, 0)
        _shade(acell, "D93B1A")
        _remove_cell_margins(acell, top=80, bottom=80, left=160, right=160)
        acell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        ap = acell.paragraphs[0]
        _clear_paragraph_spacing(ap)
        _add_run(ap, f"!  CRITICAL — {_prod_str} cannot crawl this site.  ",
                 bold=True, size=10, color=WHITE)
        _add_run(ap, f"Blocked: {_uas}. AI answers won't cite pages the crawlers can't fetch.",
                 size=8, color=WHITE)
        doc.add_paragraph()  # spacer

    # ═══════════════════════════════════════════════════════════════
    # 2. HERO HEADLINE
    # ═══════════════════════════════════════════════════════════════
    if "OVERVIEW:" in exec_summary.upper():
        ov = exec_summary.split("|")[0]
        ci = ov.find(":")
        intro_raw = clean(ov[ci+1:].strip()) if ci != -1 else clean(ov)
    else:
        intro_raw = clean(exec_summary)

    hero_p = doc.add_paragraph()
    _clear_paragraph_spacing(hero_p)
    _add_run(hero_p, "Is your site ready for the ", bold=True, size=22, color=DARK)
    _add_run(hero_p, "AI search era?", bold=True, italic=True, size=22, color=RED)

    intro_p = doc.add_paragraph()
    _clear_paragraph_spacing(intro_p)
    intro_p.paragraph_format.space_after = Pt(6)
    _add_run(intro_p, f"We audited {domain} the way ChatGPT, Perplexity, Gemini and Claude see it. ",
             bold=True, size=9, color=DARK)
    _add_run(intro_p, intro_raw, size=9, color=DARK)

    # ═══════════════════════════════════════════════════════════════
    # 3. SCORE BOX
    # ═══════════════════════════════════════════════════════════════
    verdict_text = ""
    if "VERDICT:" in exec_summary.upper():
        for part in exec_summary.split("|"):
            if "VERDICT:" in part.upper():
                ci = part.find(":")
                verdict_text = clean(part[ci+1:].strip()) if ci != -1 else ""
                break

    score_tbl = doc.add_table(rows=1, cols=2)
    score_tbl.autofit = False
    score_tbl.columns[0].width = Cm(4)
    score_tbl.columns[1].width = Cm(14.5)

    num_cell = score_tbl.cell(0, 0)
    txt_cell = score_tbl.cell(0, 1)
    _shade(num_cell, LIGHT_HEX)
    _shade(txt_cell, LIGHT_HEX)
    num_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    txt_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    num_p = num_cell.paragraphs[0]
    num_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _clear_paragraph_spacing(num_p)
    _add_run(num_p, str(avg), bold=True, size=44, color=RED)
    _add_run(num_p, "/90", size=14, color=GREY)

    lbl_p = txt_cell.paragraphs[0]
    _clear_paragraph_spacing(lbl_p)
    _add_run(lbl_p, "AVERAGE PAGE SCORE\n", size=7, color=GREY)
    _add_run(lbl_p, "A solid foundation. ", bold=True, size=14, color=DARK)
    _add_run(lbl_p, "A clear AI gap.", bold=True, size=14, color=RED)
    if verdict_text:
        vp = txt_cell.add_paragraph()
        _clear_paragraph_spacing(vp)
        _add_run(vp, verdict_text, size=8, color=GREY)

    doc.add_paragraph()  # spacer

    # ═══════════════════════════════════════════════════════════════
    # 4. DIMENSION BADGES (9-cell coloured strip)
    # ═══════════════════════════════════════════════════════════════
    dims_tbl = doc.add_table(rows=1, cols=9)
    dims_tbl.autofit = False
    for c in dims_tbl.columns:
        c.width = Cm(2.05)
    for i, (dk, dl) in enumerate(zip(dim_keys, dim_labels)):
        cell = dims_tbl.cell(0, i)
        s = dim_avg.get(dk, 0) or 0
        _shade(cell, dim_hex(s))
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _clear_paragraph_spacing(p)
        _add_run(p, dl, bold=True, size=6, color=WHITE)
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _clear_paragraph_spacing(p2)
        _add_run(p2, str(s), bold=True, size=13, color=WHITE)
        _add_run(p2, "/10", size=7, color=WHITE)

    doc.add_paragraph()  # spacer

    # ═══════════════════════════════════════════════════════════════
    # 5. WHAT'S WORKING / WHAT'S HOLDING BACK  — side by side
    # ═══════════════════════════════════════════════════════════════
    def _fill_side(cell, header_text, header_hex, icon, icon_color, items):
        # Header row
        first = cell.paragraphs[0]
        _clear_paragraph_spacing(first)
        _shade(cell, "FFFFFF")
        # Sub-table inside the outer cell for header + items
        sub = cell.add_table(rows=4, cols=1)
        sub.autofit = False
        try:
            sub.columns[0].width = Cm(9)
        except Exception:
            pass
        hdr = sub.cell(0, 0)
        _shade(hdr, header_hex)
        hp = hdr.paragraphs[0]
        _clear_paragraph_spacing(hp)
        _add_run(hp, header_text, bold=True, size=9, color=WHITE)
        for i, it in enumerate((items + [{}, {}, {}])[:3]):
            body = sub.cell(i + 1, 0)
            _shade(body, "F9F9F9")
            _remove_cell_margins(body, top=80, bottom=80, left=140, right=140)
            bp = body.paragraphs[0]
            _clear_paragraph_spacing(bp)
            pt  = clean(it.get("point", "")) if it else ""
            det = clean(it.get("detail", "")) if it else ""
            if pt:
                _add_run(bp, f"{icon}  ", bold=True, size=9, color=icon_color)
                _add_run(bp, pt, bold=True, size=9, color=DARK)
                if det:
                    dp = body.add_paragraph()
                    _clear_paragraph_spacing(dp)
                    _add_run(dp, det, size=8, color=RGBColor(0x4A, 0x4A, 0x4A))
            else:
                _add_run(bp, " ", size=9)
        # Remove the empty first paragraph we started with
        # (python-docx always seeds a cell with one paragraph)
        p0 = cell.paragraphs[0]
        if p0.text == "" and len(cell.paragraphs) > 1:
            p0._element.getparent().remove(p0._element)

    sides_tbl = doc.add_table(rows=1, cols=2)
    sides_tbl.autofit = False
    sides_tbl.columns[0].width = Cm(9)
    sides_tbl.columns[1].width = Cm(9)

    _fill_side(sides_tbl.cell(0, 0), "✚  WHAT'S WORKING",
               "27AE60", "✚", GREEN, working)
    _fill_side(sides_tbl.cell(0, 1), "!  WHAT'S HOLDING YOU BACK",
               "D93B1A", "!", RED, holding)

    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════
    # 6. THREE QUICK WINS
    # ═══════════════════════════════════════════════════════════════
    label_p = doc.add_paragraph()
    _clear_paragraph_spacing(label_p)
    _add_run(label_p, "THREE MOVES THAT MOVE THE NEEDLE\n", size=7, color=GREY)
    _add_run(label_p, "Quick wins, big impact", bold=True, size=13, color=DARK)

    wins_tbl = doc.add_table(rows=1, cols=3)
    wins_tbl.autofit = False
    for c in wins_tbl.columns:
        c.width = Cm(6.15)
    for i, w in enumerate((wins + [{}, {}, {}])[:3]):
        cell = wins_tbl.cell(0, i)
        _remove_cell_margins(cell, top=80, bottom=80, left=140, right=140)
        num    = clean(w.get("number", "")) if w else str(i + 1)
        title  = clean(w.get("title", ""))  if w else ""
        detail = clean(w.get("detail", "")) if w else ""

        np = cell.paragraphs[0]
        _clear_paragraph_spacing(np)
        _add_run(np, num or str(i + 1), bold=True, size=22, color=RED)

        if title:
            tp = cell.add_paragraph()
            _clear_paragraph_spacing(tp)
            _add_run(tp, title, bold=True, size=9, color=DARK)
        if detail:
            dp = cell.add_paragraph()
            _clear_paragraph_spacing(dp)
            _add_run(dp, detail, size=8, color=RGBColor(0x4A, 0x4A, 0x4A))
        if i < 2:
            _cell_border(cell, "right", size_eighths=4, color="DDDDDD")

    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════
    # 6b. AI BOT ACCESS — compact strip (new)
    # ═══════════════════════════════════════════════════════════════
    if bot_access and bot_access.get("bots"):
        summary = bot_access.get("summary", {})
        ba_hdr = doc.add_paragraph()
        _clear_paragraph_spacing(ba_hdr)
        _add_run(ba_hdr, "AI BOT ACCESS  ", size=7, color=GREY)
        _add_run(ba_hdr, f"{summary.get('allowed', 0)} allowed  ", bold=True, size=8, color=GREEN)
        _add_run(ba_hdr, f"·  {summary.get('partial', 0)} partial  ", bold=True, size=8, color=AMBER)
        _add_run(ba_hdr, f"·  {summary.get('blocked', 0)} blocked  ", bold=True, size=8, color=RED2)
        _add_run(ba_hdr, f"of {summary.get('total', 0)} AI bots checked", size=7, color=GREY)

        blocked_bots = [b for b in bot_access["bots"] if b.get("verdict") == "blocked"]
        if blocked_bots:
            bp = doc.add_paragraph()
            _clear_paragraph_spacing(bp)
            _add_run(bp, "Blocked: ", bold=True, size=8, color=DARK)
            _add_run(bp, ", ".join(f"{b['user_agent']} ({b['vendor']})" for b in blocked_bots[:6]),
                     size=8, color=DARK)
        doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════
    # 7. CTA BAR
    # ═══════════════════════════════════════════════════════════════
    cta_tbl = doc.add_table(rows=1, cols=2)
    cta_tbl.autofit = False
    cta_tbl.columns[0].width = Cm(9.5)
    cta_tbl.columns[1].width = Cm(9)
    left = cta_tbl.cell(0, 0)
    right = cta_tbl.cell(0, 1)
    _shade(right, LIGHT_HEX)
    _cell_border(left, "top", size_eighths=16, color="D93B1A")
    _cell_border(right, "top", size_eighths=16, color="D93B1A")
    left.vertical_alignment  = WD_ALIGN_VERTICAL.CENTER
    right.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    lp = left.paragraphs[0]
    _clear_paragraph_spacing(lp)
    _add_run(lp, "We'll walk your team through\nevery finding. ",
             bold=True, size=11, color=DARK)
    _add_run(lp, "No obligation.", bold=True, italic=True, size=11, color=RED)

    rp = right.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _clear_paragraph_spacing(rp)
    _add_run(rp, "BOOK A SESSION\n", size=7, color=GREY)
    _add_run(rp, "hello@summitmedia.com", bold=True, size=12, color=DARK)

    # ── Save & return bytes ─────────────────────────────────────────
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', Helvetica, Arial, sans-serif; }}
  .summit-header {{ background: {SUMMIT_RED}; padding: 1.2rem 1.5rem; border-radius: 8px;
    color: white; margin-bottom: 1.5rem; }}
  .summit-header h1 {{ margin: 0; font-size: 1.4rem; font-weight: 700; }}
  .summit-header p {{ margin: 0; font-size: 0.85rem; opacity: 0.85; }}
  .score-card {{ background: {SUMMIT_LIGHT}; border-radius: 8px; padding: 1rem;
    text-align: center; }}
  .score-big {{ font-size: 3rem; font-weight: 700; color: {SUMMIT_RED}; }}
  .dim-badge {{ display: inline-block; border-radius: 4px; padding: 4px 8px;
    font-size: 0.75rem; font-weight: 700; color: white; margin: 2px; }}
  .stButton > button {{ background: {SUMMIT_RED}; color: white; border: none;
    border-radius: 6px; font-weight: 600; padding: 0.6rem 1.4rem; }}
  .stButton > button:hover {{ background: #b83015; }}
  .stDownloadButton > button {{ background: {SUMMIT_RED}; color: white; border: none;
    border-radius: 6px; font-weight: 600; }}
  .stDownloadButton > button:hover {{ background: #b83015; }}
</style>
<div class="summit-header">
  <h1>🔍 Summit · AI Visibility Audit Tool</h1>
  <p>Audit any website the way ChatGPT, Perplexity, Gemini and Claude see it. Prepared by Summit.</p>
</div>
""", unsafe_allow_html=True)

# ─── Inputs ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    domain = st.text_input("Domain to audit", placeholder="e.g. example.com")
with col2:
    month_year = st.text_input("Audit month / year", value=datetime.now().strftime("%B %Y"))

st.markdown("**Additional pages to audit** (up to 3 internal URLs, one per line)")
extra_raw = st.text_area("", placeholder="https://example.com/category\nhttps://example.com/product\nhttps://example.com/blog", height=80)
extra_urls = [u for u in extra_raw.strip().splitlines() if u.strip()]

with st.expander("⚠️ Site blocked by bot protection? Paste HTML manually"):
    st.markdown(
        "Some sites (Cloudflare, Akamai etc.) block automated fetches. "
        "If the audit shows no content, open the page in Chrome, press **Ctrl+U** "
        "to view source, copy all, and paste below. You can paste up to 4 pages."
    )
    paste_entries = []
    for _i in range(4):
        _label = "Homepage" if _i == 0 else f"Page {_i + 1}"
        _pc1, _pc2 = st.columns([1, 2])
        with _pc1:
            _u = st.text_input(f"URL ({_label})", key=f"paste_url_{_i}",
                               placeholder="https://example.com/" if _i == 0 else f"https://example.com/page-{_i+1}")
        with _pc2:
            _h = st.text_area(f"HTML ({_label})", key=f"paste_html_{_i}",
                              height=100, placeholder="<!DOCTYPE html>..." if _i == 0 else "")
        if _u.strip() and _h.strip():
            paste_entries.append((_u.strip(), _h.strip()))

run = st.button("🚀 Run Audit", use_container_width=True)

# ── Run audit and store everything in session_state ────────────────────────
if run:
    if not domain:
        st.error("Please enter a domain.")
        st.stop()

    model = get_gemini_client()

    # Build pasted HTML dict from all filled-in entries
    pasted_html = {url: html for url, html in paste_entries}

    fetch_status = st.empty()
    fetch_status.info("Fetching pages…")

    pages = fetch_pages(domain, extra_urls, pasted_html or None)

    # Show what actually happened per URL and separate blocked from usable
    fetch_log = []
    blocked_urls = []
    for url, signals in pages.items():
        if "FETCH_BLOCKED" in signals:
            fetch_log.append(f"🚫 **{url}** — blocked, cannot be scored")
            blocked_urls.append(url)
        elif "FETCH_FAILED" in signals:
            fetch_log.append(f"❌ **{url}** — fetch failed")
            blocked_urls.append(url)
        elif "Wayback Machine" in signals:
            fetch_log.append(f"🗄️ **{url}** — retrieved via Wayback Machine cache")
        elif "manually provided" in signals:
            fetch_log.append(f"📋 **{url}** — using pasted HTML")
        elif "JS_SHELL" in signals:
            fetch_log.append(f"⚠️ **{url}** — JS-rendered site (shell only)")
        else:
            fetch_log.append(f"✅ **{url}** — fetched successfully")

    # Hard gate: if any URLs are blocked, halt before Gemini
    if blocked_urls:
        fetch_status.error(
            f"❌ **{len(blocked_urls)} of {len(pages)} pages could not be fetched.**\n\n"
            + "\n".join(fetch_log)
            + "\n\n---\n\n"
            + "**These pages appear to be blocked by bot protection (Cloudflare, Akamai, etc.) "
            + "or returned an empty response.**\n\n"
            + "To continue, get the raw HTML for the blocked URLs:\n"
            + "1. Open each blocked URL in your browser\n"
            + "2. Right-click, then **View Page Source** (or press Ctrl+U / Cmd+Option+U)\n"
            + "3. Select all (Ctrl+A / Cmd+A), copy (Ctrl+C / Cmd+C)\n"
            + "4. Open the **'Site blocked?'** expander above and paste the HTML against the URL\n"
            + "5. Click **Run audit** again\n\n"
            + "Scoring these pages without their real HTML would produce false results, "
            + "so the audit has been stopped."
        )
        st.stop()

    fetch_status.success(
        f"Fetched {len(pages)} page(s). Running Gemini audit…\n\n" +
        "\n".join(fetch_log)
    )

    # Compute total signal size for diagnostics — helps user understand
    # if their pages are unusually large.
    total_signal_chars = sum(len(s) for s in pages.values())
    largest_page = max(pages.items(), key=lambda kv: len(kv[1]))

    with st.spinner("Analysing with Gemini — this takes 20–40 seconds…"):
        try:
            audit = run_audit(model, pages)
        except Exception as e:
            st.error(f"❌ Gemini audit failed: {type(e).__name__}: {e}")
            # Diagnostics — help user understand WHY
            with st.expander("🔧 Diagnostic details (click to expand)"):
                st.markdown(f"**Total signal size sent to Gemini:** {total_signal_chars:,} characters")
                st.markdown("**Per-page signal sizes:**")
                for _url, _sig in pages.items():
                    over_cap = " ⚠️ over per-page cap" if len(_sig) > 40_000 else ""
                    st.markdown(f"- `{_url}` — {len(_sig):,} chars{over_cap}")
                st.markdown(
                    "**Common causes of Gemini audit failure:**\n"
                    "- Large PLPs with runaway product listings (try pasting a smaller sample page instead)\n"
                    "- Content that triggered Gemini's safety filters\n"
                    "- Temporary Gemini API issue — try again in a moment\n"
                    "- API rate limit or quota exhausted (check your Gemini console)"
                )
            st.stop()

    # ── AI bot access + robots.txt check ─────────────────────────────────
    # Runs after the Gemini audit so a failure here can't kill the audit.
    # Attached to the audit dict so it flows into every downstream output.
    with st.spinner("Checking AI bot access & robots.txt…"):
        try:
            audit["bot_access"] = check_ai_bot_access(domain)
        except Exception as e:
            audit["bot_access"] = {"error": f"{type(e).__name__}: {e}", "bots": [],
                                    "summary": {"allowed": 0, "partial": 0, "blocked": 0, "total": 0}}

    # Pre-generate all download files while we have the data
    with st.spinner("Building Word document…"):
        try:
            docx_bytes = build_docx(audit, month_year)
        except Exception as e:
            docx_bytes = None
            st.warning(f"Word doc error: {e}")

    with st.spinner("Building one-pager PDF…"):
        try:
            pdf_bytes = build_onepager(audit, month_year)
        except Exception as e:
            pdf_bytes = None
            st.warning(f"PDF error: {e}")

    with st.spinner("Building one-pager Word doc…"):
        try:
            onepager_docx_bytes = build_onepager_docx(audit, month_year)
        except Exception as e:
            onepager_docx_bytes = None
            st.warning(f"One-pager Word error: {e}")

    # Store everything — survives download-button reruns
    st.session_state["audit"]              = audit
    st.session_state["month_year"]         = month_year
    st.session_state["docx_bytes"]         = docx_bytes
    st.session_state["pdf_bytes"]          = pdf_bytes
    st.session_state["onepager_docx_bytes"] = onepager_docx_bytes

# ── Display results from session_state (persists across reruns) ────────────
if "audit" in st.session_state:
    audit      = st.session_state["audit"]
    month_year = st.session_state["month_year"]
    docx_bytes = st.session_state["docx_bytes"]
    pdf_bytes  = st.session_state["pdf_bytes"]

    company  = audit.get("company_name", domain)
    avg      = round(audit.get("average_score", 0))
    dim_avg  = audit.get("dimension_averages", {})
    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    st.markdown(f"## 📊 Results: {company}")

    # ── Critical AI bot access alert (top-of-page banner) ─────────────────
    # If any of ChatGPT / Claude / Perplexity / Gemini can't crawl the site,
    # nothing else in the audit matters as much as that fact. Show it first.
    _blocked_critical = get_blocked_critical_bots(audit.get("bot_access", {}))
    if _blocked_critical:
        products = sorted({b["product"] for b in _blocked_critical})
        products_str = ", ".join(products[:-1]) + (f" and {products[-1]}" if len(products) > 1 else products[0])
        bullet_lines = "\n".join(
            f"- **{b['product']}** — `{b['user_agent']}` blocked ({b['robots_evidence'] if b['robots_status'].startswith('blocked') else 'live fetch returned ' + str(b['http_code'])})"
            for b in _blocked_critical
        )
        st.error(
            f"🚫 **Critical AI crawler{'s' if len(_blocked_critical) > 1 else ''} blocked — "
            f"{products_str} cannot access this site.**\n\n"
            f"This overrides everything else in the audit. Even a perfect score won't earn "
            f"AI citations if the crawlers that feed those products can't fetch your pages.\n\n"
            f"{bullet_lines}\n\n"
            f"See the **AI Bot Access** section below for the full breakdown and remediation."
        )

    # Score overview
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"""
        <div class="score-card">
          <div style="font-size:0.7rem;font-weight:600;color:{SUMMIT_GREY};letter-spacing:1px">AVERAGE PAGE SCORE</div>
          <div class="score-big">{avg}</div>
          <div style="color:{SUMMIT_GREY}">/90</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        n_pages = len(audit.get("pages", []))
        page_label = f"Average across {n_pages} page{'s' if n_pages != 1 else ''}"
        badges = f'<div style="font-size:0.7rem;color:{SUMMIT_GREY};margin-bottom:4px">{page_label}</div>'
        for dk, dl in zip(dim_keys, dim_labels):
            s = dim_avg.get(dk, 0)
            badges += f'<span class="dim-badge" style="background:{score_color(s)}">{dl}: {s}/10</span>'
        st.markdown(f'<div style="padding:1rem">{badges}</div>', unsafe_allow_html=True)
        summary_full = audit.get('executive_summary','')
        # Render the structured pipe-delimited format nicely in Streamlit
        if '|' in summary_full:
            sections = [s.strip() for s in summary_full.split('|') if s.strip()]
            for sec in sections:
                colon = sec.find(':')
                if colon != -1:
                    label = sec[:colon].strip()
                    body  = sec[colon+1:].strip()
                    # Bullet lines start with *
                    if '*' in body:
                        bullets = [b.strip() for b in body.split('*') if b.strip()]
                        st.markdown(f"**{label}**")
                        for b in bullets:
                            st.markdown(f"- {b}")
                    else:
                        st.markdown(f"**{label}:** {body}")
                else:
                    st.markdown(sec)
        else:
            st.markdown(f"**Executive summary:** {summary_full[:300]}{'...' if len(summary_full)>300 else ''}")
            if len(summary_full) > 300:
                with st.expander("Read full summary"):
                    st.markdown(summary_full)

    # Page tabs
    pages_data = audit.get("pages", [])
    if pages_data:
        tabs = st.tabs([p.get("title", f"Page {i+1}") for i, p in enumerate(pages_data)])
        for tab, page in zip(tabs, pages_data):
            with tab:
                st.markdown(f"**Score:** {page.get('score',0)}/90 &nbsp;|&nbsp; *{page.get('verdict','')}*")
                st.info(page.get("headline_finding",""))
                dims = page.get("dimensions", {})
                rows = []
                for dk, dl in zip(dim_keys, dim_labels):
                    d = dims.get(dk, {})
                    rows.append({"Dimension": dl, "Score": f"{d.get('score',0)}/10", "Detail": d.get("detail","")})
                st.table(rows)
                sfinds = page.get("specific_findings", [])
                if sfinds:
                    st.markdown("**Specific findings:**")
                    for sf in sfinds:
                        st.markdown(f"- {sf}")

    # Recommendations
    recs = audit.get("recommendations", [])
    if recs:
        st.markdown("### 📋 Priority Recommendations")
        st.table([{
            "Priority": r.get("priority",""),
            "Action": r.get("action","")[:120],
            "Impact": r.get("impact",""),
            "Effort": r.get("effort",""),
            "Owner": r.get("owner",""),
        } for r in recs])

    # ── AI bot access (robots.txt + live WAF check) ───────────────────────
    ba = audit.get("bot_access") or {}
    if ba and ba.get("bots"):
        st.markdown("### 🤖 AI Bot Access & robots.txt")
        s = ba.get("summary", {})
        # Traffic-light summary row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Allowed",       s.get("allowed", 0))
        m2.metric("Partial",       s.get("partial", 0))
        m3.metric("Blocked",       s.get("blocked", 0))
        m4.metric("Bots checked",  s.get("total", 0))

        if ba.get("robots_txt_found"):
            st.caption(f"✅ robots.txt found at `{ba.get('robots_txt_url','')}`")
        else:
            err = ba.get("robots_txt_error", "")
            st.caption(f"⚠️ No robots.txt found. {err}")

        # Per-bot table
        verdict_icon = {"allowed": "✅ Allowed", "partial": "🟡 Partial", "blocked": "🚫 Blocked"}
        rob_label = {
            "blocked_all":     "Blocked (all)",
            "blocked_partial": "Blocked (paths)",
            "allowed":         "Allowed",
            "not_specified":   "Not specified",
            "no_robots":       "No robots.txt",
        }
        live_label = {
            "allowed":      "200 OK",
            "blocked":      "Blocked",
            "server_error": "Server error",
            "timeout":      "Timeout",
            "error":        "Network error",
            "other":        "Other",
            "unknown":      "Unknown",
        }
        st.table([{
            "Bot":         b["user_agent"],
            "Vendor":      b["vendor"],
            "Overall":     verdict_icon.get(b["verdict"], b["verdict"]),
            "robots.txt":  rob_label.get(b["robots_status"], b["robots_status"]),
            "Live check":  f"{live_label.get(b['live_status'], b['live_status'])}"
                           + (f" (HTTP {b['http_code']})" if b['http_code'] else ""),
            "Purpose":     b["purpose"],
        } for b in ba["bots"]])

        if ba.get("robots_txt_content"):
            with st.expander("View robots.txt contents"):
                st.code(ba["robots_txt_content"], language="text")

    # ── Download buttons — data already in memory, no recompute ───────────
    st.markdown("---")
    st.markdown("### 📥 Download Outputs")
    col_d, col_p, col_op = st.columns(3)

    slug = company.lower().replace(' ', '-').replace('.', '')
    onepager_docx_bytes = st.session_state.get("onepager_docx_bytes")

    with col_d:
        if docx_bytes:
            st.download_button(
                "⬇️ Full Audit (.docx)",
                data=docx_bytes,
                file_name=f"summit-ai-audit-{slug}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_docx",
            )
        else:
            st.error("Full audit Word doc could not be generated.")

    with col_p:
        if pdf_bytes:
            st.download_button(
                "⬇️ One-Pager (.pdf)",
                data=pdf_bytes,
                file_name=f"summit-ai-snapshot-{slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf",
            )
        else:
            st.error("One-pager PDF could not be generated.")

    with col_op:
        if onepager_docx_bytes:
            st.download_button(
                "⬇️ One-Pager (.docx)",
                data=onepager_docx_bytes,
                file_name=f"summit-ai-snapshot-{slug}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_onepager_docx",
                help="Editable Word version of the one-pager — useful if you want to tweak copy before sending.",
            )
        else:
            st.error("One-pager Word doc could not be generated.")

    # Clear results button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear & run new audit"):
        for k in ["audit","month_year","docx_bytes","pdf_bytes","onepager_docx_bytes"]:
            st.session_state.pop(k, None)
        st.rerun()

