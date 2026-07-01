# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SCHEMA SCORER
# Drop this in above the AUDIT_PROMPT definition in app.py.
# Replaces Gemini's SCHEMA score with a code-based one that reads the raw HTML,
# parses JSON-LD, and scores against page-type expectations.
# ═══════════════════════════════════════════════════════════════════════════════

import json as _sj
import re as _sre
from bs4 import BeautifulSoup as _SBS

# ── Which schema types are appropriate for each page type ────────────────────
# Format: {page_type: (expected_types, high_value_types, required_props_per_type)}
# expected_types    = types we would expect to see (any of these = on-topic)
# high_value_types  = types that unlock rich results / AI citation on this page
# required_props    = properties a type must have to count as "well-populated"

SCHEMA_EXPECTATIONS = {
    "homepage": {
        "expected": {"Organization", "WebSite", "Corporation", "LocalBusiness"},
        "high_value": {"Organization", "WebSite"},
        "bonus": {"BreadcrumbList", "SiteNavigationElement"},
    },
    "category": {
        "expected": {"CollectionPage", "ItemList", "BreadcrumbList", "WebPage"},
        "high_value": {"ItemList", "BreadcrumbList"},
        "bonus": {"Organization", "WebSite"},
    },
    "product": {
        "expected": {"Product", "BreadcrumbList", "Offer"},
        "high_value": {"Product", "Offer"},
        "bonus": {"BreadcrumbList", "Review", "AggregateRating", "FAQPage"},
    },
    "article": {
        "expected": {"Article", "BlogPosting", "NewsArticle", "Recipe"},
        "high_value": {"Article", "BlogPosting", "NewsArticle"},
        "bonus": {"BreadcrumbList", "Person", "Organization", "FAQPage"},
    },
    "other": {
        "expected": {"WebPage", "AboutPage", "ContactPage", "FAQPage"},
        "high_value": {"WebPage", "FAQPage"},
        "bonus": {"BreadcrumbList", "Organization"},
    },
}

# ── Required properties per type to count as "well-populated" ────────────────
# Missing a required property drops the type from "well-populated" to "sparse".

REQUIRED_PROPS = {
    "Product":       {"name", "image", "description", "offers"},
    "Offer":         {"price", "priceCurrency", "availability"},
    "Article":       {"headline", "author", "datePublished"},
    "BlogPosting":   {"headline", "author", "datePublished"},
    "NewsArticle":   {"headline", "author", "datePublished"},
    "Organization":  {"name", "url"},
    "WebSite":       {"name", "url"},
    "ItemList":      {"itemListElement"},
    "BreadcrumbList": {"itemListElement"},
    "FAQPage":       {"mainEntity"},
    "Recipe":        {"name", "recipeIngredient", "recipeInstructions"},
    "Person":        {"name"},
    "AboutPage":     {"name"},
    "ContactPage":   {"name"},
    "WebPage":       {"name"},
}


def _extract_jsonld_blocks(html: str) -> list[dict]:
    """
    Extract all JSON-LD blocks from the HTML. Returns a list of dicts, one per
    valid parseable block. Broken blocks are dropped but counted separately
    (see score_schema for how invalid blocks affect the score).
    """
    soup = _SBS(html, "html.parser")
    blocks = []
    invalid_count = 0

    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    # Fallback: regex, in case BeautifulSoup missed any due to encoding oddities
    if not scripts:
        raw = _sre.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, _sre.DOTALL | _sre.IGNORECASE
        )
        for r in raw:
            txt = r.strip()
            if not txt:
                continue
            try:
                parsed = _sj.loads(txt)
                if isinstance(parsed, list):
                    blocks.extend(parsed)
                elif isinstance(parsed, dict):
                    if "@graph" in parsed and isinstance(parsed["@graph"], list):
                        blocks.extend(parsed["@graph"])
                    else:
                        blocks.append(parsed)
            except Exception:
                invalid_count += 1
        return blocks, invalid_count

    for s in scripts:
        txt = s.string or s.get_text()
        if not txt or not txt.strip():
            continue
        try:
            parsed = _sj.loads(txt.strip())
            # JSON-LD can be a single object, an array of objects, or an
            # object with @graph containing an array.
            if isinstance(parsed, list):
                blocks.extend(parsed)
            elif isinstance(parsed, dict):
                if "@graph" in parsed and isinstance(parsed["@graph"], list):
                    blocks.extend(parsed["@graph"])
                else:
                    blocks.append(parsed)
        except Exception:
            invalid_count += 1

    return blocks, invalid_count


def _get_types(block: dict) -> set[str]:
    """A JSON-LD block's @type can be a string or a list. Return as a set."""
    t = block.get("@type", "")
    if isinstance(t, list):
        return {str(x) for x in t}
    if isinstance(t, str) and t:
        return {t}
    return set()


def _is_well_populated(block: dict, block_types: set[str]) -> bool:
    """
    A block is well-populated if it has all required properties for at least
    one of its declared types. Types not in REQUIRED_PROPS are considered
    well-populated if they have any properties beyond @type and @context.
    """
    props = set(block.keys()) - {"@type", "@context", "@id"}
    for t in block_types:
        if t in REQUIRED_PROPS:
            if REQUIRED_PROPS[t].issubset(props):
                return True
        else:
            # Unknown type: any populated properties count as well-populated
            if len(props) >= 2:
                return True
    return False


def _collect_nested_types(obj, depth=0) -> set[str]:
    """Recursively walk a JSON-LD block and collect all @type values found."""
    if depth > 6:
        return set()
    found = set()
    if isinstance(obj, dict):
        t = obj.get("@type", "")
        if isinstance(t, list):
            found.update(str(x) for x in t)
        elif isinstance(t, str) and t:
            found.add(t)
        for v in obj.values():
            found.update(_collect_nested_types(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            found.update(_collect_nested_types(item, depth + 1))
    return found


def score_schema(html: str, page_type: str) -> dict:
    """
    Deterministic SCHEMA scorer. Returns a dict with:
      - score:      integer 1-10
      - detail:     human-readable finding for the audit doc
      - evidence:   structured facts the score is built from (for debugging)

    The score is derived from evidence, not the other way round. If you
    disagree with a score, look at the evidence to see what the code saw.
    """
    if page_type not in SCHEMA_EXPECTATIONS:
        page_type = "other"

    expectations = SCHEMA_EXPECTATIONS[page_type]
    expected     = expectations["expected"]
    high_value   = expectations["high_value"]
    bonus        = expectations["bonus"]

    blocks, invalid_count = _extract_jsonld_blocks(html)

    # ── Evidence gathering ──────────────────────────────────────────
    all_types = set()
    for b in blocks:
        all_types.update(_get_types(b))
        # Nested types matter for spotting AggregateRating and Review inside
        # Product blocks, but we filter out structural sub-types that are
        # implementation details of their parent (ListItem inside BreadcrumbList,
        # Question/Answer inside FAQPage, Rating inside AggregateRating).
        all_types.update(_collect_nested_types(b))

    # Filter out structural sub-types that don't reflect additional schema
    # investment, they're just parts of their parent type.
    STRUCTURAL_SUBTYPES = {
        "ListItem", "Question", "Answer", "Rating", "PostalAddress",
        "GeoCoordinates", "OpeningHoursSpecification", "ContactPoint",
        "PropertyValue", "ImageObject", "VideoObject",
    }
    all_types = all_types - STRUCTURAL_SUBTYPES

    expected_hits    = expected & all_types
    high_value_hits  = high_value & all_types
    bonus_hits       = bonus & all_types

    populated_blocks = 0
    sparse_blocks    = 0
    for b in blocks:
        types = _get_types(b)
        if not types:
            continue
        # Only count blocks that have at least one expected or bonus type
        if types & (expected | bonus):
            if _is_well_populated(b, types):
                populated_blocks += 1
            else:
                sparse_blocks += 1

    evidence = {
        "total_blocks":       len(blocks),
        "invalid_blocks":     invalid_count,
        "all_types_found":    sorted(all_types),
        "expected_types_hit": sorted(expected_hits),
        "high_value_hits":    sorted(high_value_hits),
        "bonus_hits":         sorted(bonus_hits),
        "populated_blocks":   populated_blocks,
        "sparse_blocks":      sparse_blocks,
        "page_type":          page_type,
    }

    # ── Score derivation ────────────────────────────────────────────
    # The rubric bands:
    # 1 = no JSON-LD at all
    # 2 = present but all invalid
    # 3 = valid but no appropriate types
    # 4 = one appropriate type, sparse
    # 5 = one appropriate type, populated (or two expected sparse)
    # 6 = high-value type populated OR expected + meaningful bonus (Breadcrumb/FAQ/Review)
    # 7 = high-value populated + one bonus type OR two expected + one bonus
    # 8 = high-value populated + two bonus types OR three expected all populated
    # 9 = 8 plus AggregateRating or Review or FAQPage where relevant
    # 10 = reserved, never awarded automatically

    # Which bonus types are "meaningful" (drive AI citation on this page type)
    meaningful_bonus = bonus_hits & {
        "BreadcrumbList", "FAQPage", "Review", "AggregateRating",
        "Person", "Organization"
    }

    # No blocks at all
    if len(blocks) == 0 and invalid_count == 0:
        score = 1
    # Something was there but all failed to parse
    elif len(blocks) == 0 and invalid_count > 0:
        score = 2
    # Valid blocks but nothing appropriate for the page type
    elif not expected_hits and not bonus_hits:
        score = 3
    # Only bonus types on their own (e.g. only BreadcrumbList on a homepage)
    # counts as slightly better than wrong-type but worse than expected-type
    elif not expected_hits and bonus_hits:
        score = 4
    # One expected type, sparse
    elif len(expected_hits) == 1 and populated_blocks == 0:
        score = 4
    # One expected type populated
    elif len(expected_hits) == 1 and populated_blocks >= 1 and not meaningful_bonus:
        score = 5
    # One expected populated + one meaningful bonus type
    elif len(expected_hits) == 1 and populated_blocks >= 1 and len(meaningful_bonus) == 1:
        score = 6
    # One expected populated + two or more meaningful bonuses
    elif len(expected_hits) == 1 and populated_blocks >= 1 and len(meaningful_bonus) >= 2:
        score = 7
    # Two expected types, at least one populated
    elif len(expected_hits) >= 2 and populated_blocks >= 1 and not meaningful_bonus:
        score = 6
    # Two expected types populated + meaningful bonus
    elif len(expected_hits) >= 2 and populated_blocks >= 2 and len(meaningful_bonus) >= 1:
        score = 7
    # Three+ expected types all populated + meaningful bonus
    elif len(expected_hits) >= 3 and populated_blocks >= 3:
        score = 8
    else:
        # Fallback: something's there but doesn't cleanly fit a band
        score = 4

    # Uplift: 8+ with AggregateRating or Review present
    if score >= 7 and {"Review", "AggregateRating"} & all_types:
        score = min(score + 1, 9)

    # Deduction: invalid blocks alongside valid ones is still a hygiene problem
    if invalid_count > 0 and score >= 4:
        score = max(score - 1, 3)

    # Cap: 10 is reserved and never awarded automatically
    score = min(score, 9)

    # ── Human-readable detail for the audit doc ─────────────────────
    detail = _build_detail(evidence, page_type)

    return {"score": score, "detail": detail, "evidence": evidence}


def _build_detail(ev: dict, page_type: str) -> str:
    """
    Turn the evidence dict into a British English sentence for the audit
    doc. No long dashes. Neutral, factual tone matching Summit's style.
    """
    parts = []
    if ev["total_blocks"] == 0 and ev["invalid_blocks"] == 0:
        return (
            f"No schema.org JSON-LD was detected on this {page_type} page. "
            "For AI systems this means no explicit signals about page type, "
            "entities, or relationships, so the crawler must infer everything "
            "from the surrounding HTML."
        )
    if ev["total_blocks"] == 0 and ev["invalid_blocks"] > 0:
        return (
            f"JSON-LD blocks are present on this page but all "
            f"{ev['invalid_blocks']} failed to parse. Broken structured data "
            "is worse than none: it signals neglect and is ignored by AI "
            "systems that would otherwise use it."
        )
    parts.append(
        f"{ev['total_blocks']} JSON-LD block(s) detected, declaring types: "
        f"{', '.join(ev['all_types_found']) if ev['all_types_found'] else 'none'}."
    )
    if ev["expected_types_hit"]:
        parts.append(
            f"Types appropriate for a {page_type} page: "
            f"{', '.join(ev['expected_types_hit'])}."
        )
    else:
        parts.append(
            f"None of the declared types are the ones AI systems expect on a "
            f"{page_type} page."
        )
    if ev["sparse_blocks"] > 0:
        parts.append(
            f"{ev['sparse_blocks']} block(s) are sparsely populated and lack "
            "required properties, limiting AI extraction value."
        )
    if ev["invalid_blocks"] > 0:
        parts.append(
            f"{ev['invalid_blocks']} additional block(s) failed to parse and "
            "should be repaired."
        )
    return " ".join(parts)
