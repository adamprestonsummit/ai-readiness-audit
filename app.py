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

# ─── Fetch URL HTML ───────────────────────────────────────────────────────────
def fetch_pages(domain: str, extra_urls: list[str]) -> dict[str, str]:
    """Fetch up to 4 pages: homepage + extras."""
    base = domain.rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base

    urls = [base] + [u.strip() for u in extra_urls if u.strip()][:3]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SummitAIAudit/1.0; "
            "+https://summit.co.uk)"
        )
    }
    pages = {}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            # strip script/style noise
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            pages[url] = soup.prettify()[:40000]   # cap tokens
        except Exception as e:
            pages[url] = f"[ERROR fetching {url}: {e}]"
    return pages

# ─── Gemini audit ─────────────────────────────────────────────────────────────
AUDIT_PROMPT = """
You are an expert AI visibility auditor. Audit the provided HTML pages exactly like a senior technical SEO and AI readiness consultant would.

Score EACH page 1-10 across these 8 dimensions:
1. ARIA – landmark roles, aria-labels, accessibility for AI parsers
2. SCHEMA – schema.org JSON-LD structured data presence and quality
3. HEADINGS – H1-H6 hierarchy, clarity, topic signal
4. META – title, description, canonical, Open Graph, Twitter Card
5. LINKS – internal link quality, anchor text, protocol consistency, density
6. ALT TEXT – image alt attribute quality and completeness
7. CRAWL – server-rendered static HTML vs JS dependency
8. LLM – first-hand expertise, named entities, dates, citations, authority signals

CRITICAL RULES:
- Return ONLY raw JSON. No markdown, no ```json fences, no preamble, no explanation.
- All string values must use double quotes. Never use single quotes inside JSON strings.
- Escape any double quotes inside string values with a backslash.
- Do not include trailing commas after the last item in any array or object.
- Every string value must be on a single line with no literal newlines inside strings.

Return ONLY valid JSON in exactly this structure:
{
  "company_name": "string",
  "domain": "string",
  "executive_summary": "3-4 paragraph string describing overall AI readiness",
  "average_score": number,
  "dimension_averages": {
    "aria": number, "schema": number, "headings": number, "meta": number,
    "links": number, "alt_text": number, "crawl": number, "llm": number
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
        "llm": {"score": number, "detail": "string"}
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
    {"point": "string", "detail": "string"}
  ]
}
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


def run_audit(model, pages: dict) -> dict:
    pages_text = ""
    for url, html in pages.items():
        pages_text += f"\n\n=== URL: {url} ===\n{html[:8000]}"

    response = model.generate_content(
        AUDIT_PROMPT + "\n\nPages to audit:\n" + pages_text,
        generation_config={"temperature": 0.1, "max_output_tokens": 8192},
    )
    raw = clean_json_string(response.text)

    try:
        return repair_and_parse(raw)
    except ValueError:
        # Retry: ask Gemini to fix its own output
        fix_prompt = (
            "The following text should be valid JSON but contains errors. "
            "Return ONLY the corrected JSON object with no other text, "
            "no markdown fences, no explanation:\n\n" + raw[:6000]
        )
        retry = model.generate_content(
            fix_prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 8192},
        )
        raw2 = clean_json_string(retry.text)
        return repair_and_parse(raw2)

# ─── Score colour ─────────────────────────────────────────────────────────────
def score_color(s):
    if s <= 2: return "#C0392B"   # red
    if s <= 5: return "#E67E22"   # amber
    return "#27AE60"              # green

# ─── Build Word doc ───────────────────────────────────────────────────────────
def build_docx(data: dict, month_year: str) -> bytes:
    """Write a Node.js docx-generating script and run it."""
    company  = data.get("company_name", "Client")
    domain   = data.get("domain", "")
    avg      = data.get("average_score", 0)
    dim_avg  = data.get("dimension_averages", {})
    pages    = data.get("pages", [])
    themes   = data.get("cross_cutting_themes", [])
    recs     = data.get("recommendations", [])
    summary  = data.get("executive_summary", "")

    # Escape strings for JS
    def js(s):
        return json.dumps(str(s))

    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]
    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]

    def score_color_hex(s):
        if s <= 2: return "C0392B"
        if s <= 5: return "E67E22"
        return "27AE60"

    # Build page sections JS
    page_sections_js = ""
    for i, p in enumerate(pages, 1):
        url     = p.get("url","")
        title   = p.get("title", f"Page {i}")
        pscore  = p.get("score", 0)
        verdict = p.get("verdict","")
        hf      = p.get("headline_finding","")
        dims    = p.get("dimensions", {})
        sfinds  = p.get("specific_findings", [])

        # dimension table rows
        dim_rows_js = ""
        for dk, dl in zip(dim_keys, dim_labels):
            d = dims.get(dk, {})
            s = d.get("score", 0)
            det = d.get("detail","")
            col = score_color_hex(s)
            dim_rows_js += f"""
      new TableRow({{ children: [
        new TableCell({{ borders, width: {{ size: 1200, type: WidthType.DXA }},
          shading: {{ fill: "{col}", type: ShadingType.CLEAR }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(dl)}, bold: true, color: "FFFFFF", font: "Arial", size: 18 }})] }})] }}),
        new TableCell({{ borders, width: {{ size: 600, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ alignment: AlignmentType.CENTER, children: [new TextRun({{ text: {js(str(s)+"/10")}, bold: true, font: "Arial", size: 20 }})] }})] }}),
        new TableCell({{ borders, width: {{ size: 7160, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(det)}, font: "Arial", size: 18 }})] }})] }}),
      ]}})"""

        # specific findings
        sf_js = ""
        for sf in sfinds:
            sf_js += f"""
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }},
        children: [new TextRun({{ text: {js(sf)}, font: "Arial", size: 20 }})] }}),"""

        page_sections_js += f"""
    // ── Page {i} ──
    new Paragraph({{ heading: HeadingLevel.HEADING_1,
      children: [new TextRun({{ text: {js(f"PAGE {i}: {title.upper()}")}, font: "Arial", size: 28, bold: true, color: "D93B1A" }})] }}),
    new Paragraph({{ children: [new TextRun({{ text: "URL: ", bold: true, font: "Arial", size: 20 }}),
      new TextRun({{ text: {js(url)}, font: "Arial", size: 20, color: "D93B1A" }})] }}),
    new Paragraph({{ children: [new TextRun({{ text: {js(f"Total score: {pscore}/80")}, bold: true, font: "Arial", size: 20 }})] }}),
    new Paragraph({{ spacing: {{ before: 160 }},
      border: {{ left: {{ style: BorderStyle.SINGLE, size: 20, color: "D93B1A" }} }},
      indent: {{ left: 360 }},
      children: [new TextRun({{ text: {js(hf)}, font: "Arial", size: 20, italics: true }})] }}),
    new Paragraph({{ heading: HeadingLevel.HEADING_2,
      children: [new TextRun({{ text: "Dimension Breakdown", font: "Arial", size: 24, bold: true }})] }}),
    new Table({{
      width: {{ size: 8960, type: WidthType.DXA }},
      columnWidths: [1200, 600, 7160],
      rows: [{dim_rows_js}
      ]
    }}),
    new Paragraph({{ heading: HeadingLevel.HEADING_2,
      children: [new TextRun({{ text: "Specific Findings", font: "Arial", size: 24, bold: true }})] }}),
    {sf_js if sf_js else "new Paragraph({ children: [new TextRun({ text: 'No specific findings recorded.', font: 'Arial', size: 20 })] }),"}
    new Paragraph({{ children: [new PageBreak()] }}),"""

    # Cross-cutting themes JS
    themes_js = ""
    for t in themes:
        title_t = t.get("title","")
        detail_t = t.get("detail","")
        themes_js += f"""
    new Paragraph({{ heading: HeadingLevel.HEADING_2,
      children: [new TextRun({{ text: {js(title_t)}, font: "Arial", size: 24, bold: true }})] }}),
    new Paragraph({{ spacing: {{ before: 80, after: 160 }},
      children: [new TextRun({{ text: {js(detail_t)}, font: "Arial", size: 20 }})] }}),"""

    # Recommendations table rows JS
    rec_rows_js = """
      new TableRow({ tableHeader: true, children: [
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 600, type: WidthType.DXA },
          margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "PRI.", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 4200, type: WidthType.DXA },
          margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "ACTION", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 1800, type: WidthType.DXA },
          margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "IMPACT", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 1500, type: WidthType.DXA },
          margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "EFFORT", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 1220, type: WidthType.DXA },
          margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "OWNER", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
      ]}),"""

    for i, r in enumerate(recs):
        row_fill = "F5F4F2" if i % 2 == 0 else "FFFFFF"
        rec_rows_js += f"""
      new TableRow({{ children: [
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 600, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(r.get("priority",""))}, bold: true, font: "Arial", size: 18, color: "D93B1A" }})] }})] }}),
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 4200, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(r.get("action",""))}, font: "Arial", size: 18 }})] }})] }}),
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 1800, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(r.get("impact",""))}, font: "Arial", size: 18 }})] }})] }}),
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 1500, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(r.get("effort",""))}, font: "Arial", size: 18 }})] }})] }}),
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 1220, type: WidthType.DXA }},
          margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(r.get("owner",""))}, font: "Arial", size: 18 }})] }})] }}),
      ] }}),"""

    # Scorecard table rows JS
    scorecard_rows_js = """
      new TableRow({ tableHeader: true, children: [
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 3500, type: WidthType.DXA }, margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "PAGE", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 900, type: WidthType.DXA }, margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "SCORE", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
        new TableCell({ borders, shading: { fill: "D93B1A", type: ShadingType.CLEAR },
          width: { size: 4960, type: WidthType.DXA }, margins: { top:80, bottom:80, left:120, right:120 },
          children: [new Paragraph({ children: [new TextRun({ text: "VERDICT", bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })] }),
      ]}),"""

    for i, p in enumerate(pages):
        row_fill = "F5F4F2" if i % 2 == 0 else "FFFFFF"
        scorecard_rows_js += f"""
      new TableRow({{ children: [
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 3500, type: WidthType.DXA }}, margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(p.get("title",""))}, bold: true, font: "Arial", size: 18 }})] }})] }}),
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 900, type: WidthType.DXA }}, margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ alignment: AlignmentType.CENTER, children: [new TextRun({{ text: {js(str(p.get("score",0))+"/80")}, bold: true, font: "Arial", size: 18 }})] }})] }}),
        new TableCell({{ borders, shading: {{ fill: {js(row_fill)}, type: ShadingType.CLEAR }},
          width: {{ size: 4960, type: WidthType.DXA }}, margins: {{ top:80, bottom:80, left:120, right:120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: {js(p.get("verdict",""))}, font: "Arial", size: 18 }})] }})] }}),
      ] }}),"""

    # Dimension averages scorecard rows
    dim_avg_row1 = ""
    for dk, dl in zip(dim_keys, dim_labels):
        s = dim_avg.get(dk, 0)
        col = score_color_hex(s)
        dim_avg_row1 += f"""
        new TableCell({{ borders, shading: {{ fill: "{col}", type: ShadingType.CLEAR }},
          width: {{ size: 1170, type: WidthType.DXA }}, margins: {{ top:80, bottom:80, left:60, right:60 }},
          children: [
            new Paragraph({{ alignment: AlignmentType.CENTER, children: [new TextRun({{ text: {js(dl)}, font: "Arial", size: 14, color: "FFFFFF", bold: true }})] }}),
            new Paragraph({{ alignment: AlignmentType.CENTER, children: [new TextRun({{ text: {js(str(s)+"/10")}, font: "Arial", size: 24, bold: true, color: "FFFFFF" }})] }}),
          ] }}),"""

    logo_path_js = json.dumps(LOGO_PATH)

    script = f"""
const fs = require('fs');
const path = require('path');
const {{
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  LevelFormat, ExternalHyperlink
}} = require('docx');

const logoData = fs.readFileSync({logo_path_js});
const border = {{ style: BorderStyle.SINGLE, size: 1, color: "DDDDDD" }};
const borders = {{ top: border, bottom: border, left: border, right: border }};
const noBorder = {{ style: BorderStyle.NONE }};
const noBorders = {{ top: noBorder, bottom: noBorder, left: noBorder, right: noBorder }};

const doc = new Document({{
  numbering: {{
    config: [
      {{ reference: "bullets", levels: [{{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: {{ paragraph: {{ indent: {{ left: 720, hanging: 360 }} }} }} }}] }},
    ]
  }},
  styles: {{
    default: {{ document: {{ run: {{ font: "Arial", size: 20 }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: {{ size: 32, bold: true, font: "Arial", color: "1A1A1A" }},
        paragraph: {{ spacing: {{ before: 320, after: 160 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: {{ size: 24, bold: true, font: "Arial", color: "1A1A1A" }},
        paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 1 }} }},
    ]
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 11906, height: 16838 }},
        margin: {{ top: 1000, right: 1000, bottom: 1000, left: 1000 }}
      }}
    }},
    headers: {{
      default: new Header({{
        children: [
          new Table({{
            width: {{ size: 9906, type: WidthType.DXA }},
            columnWidths: [1200, 8706],
            rows: [
              new TableRow({{ children: [
                new TableCell({{ borders: noBorders, width: {{ size: 1200, type: WidthType.DXA }},
                  children: [new Paragraph({{ children: [
                    new ImageRun({{ data: logoData, transformation: {{ width: 50, height: 50 }}, type: "png" }})
                  ] }})] }}),
                new TableCell({{ borders: noBorders, width: {{ size: 8706, type: WidthType.DXA }},
                  verticalAlign: VerticalAlign.CENTER,
                  children: [new Paragraph({{ alignment: AlignmentType.RIGHT, children: [
                    new TextRun({{ text: "AI VISIBILITY AUDIT · {company.upper()} · {month_year.upper()}", font: "Arial", size: 16, color: "6B6B6B" }})
                  ] }})] }}),
              ] }})
            ]
          }}),
          new Paragraph({{ border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "D93B1A" }} }}, children: [] }}),
        ]
      }})
    }},
    footers: {{
      default: new Footer({{
        children: [
          new Paragraph({{ border: {{ top: {{ style: BorderStyle.SINGLE, size: 4, color: "D93B1A" }} }},
            children: [] }}),
          new Paragraph({{ children: [
            new TextRun({{ text: "SUMMITMEDIA.CO.UK", font: "Arial", size: 16, color: "6B6B6B" }}),
            new TextRun({{ text: "\\t\\tPREPARED BY SUMMIT · AI VISIBILITY PRACTICE", font: "Arial", size: 16, color: "6B6B6B" }}),
            new TextRun({{ text: "\\t\\t", font: "Arial", size: 16 }}),
            new TextRun({{ children: [new PageNumber()], font: "Arial", size: 16, color: "6B6B6B" }}),
          ] }}),
        ]
      }})
    }},
    children: [
      // ── Cover ──
      new Paragraph({{ spacing: {{ before: 400 }}, children: [
        new TextRun({{ text: "AI VISIBILITY AUDIT", font: "Arial", size: 56, bold: true, color: "1A1A1A" }})
      ] }}),
      new Paragraph({{ children: [
        new TextRun({{ text: {js(company)}, font: "Arial", size: 40, bold: true, color: "D93B1A" }})
      ] }}),
      new Paragraph({{ spacing: {{ after: 320 }}, children: [
        new TextRun({{ text: "Technical readiness for the AI search era", font: "Arial", size: 24, italics: true, color: "6B6B6B" }})
      ] }}),
      new Table({{
        width: {{ size: 4000, type: WidthType.DXA }},
        columnWidths: [1800, 2200],
        rows: [
          new TableRow({{ children: [
            new TableCell({{ borders: noBorders, width: {{ size: 1800, type: WidthType.DXA }},
              shading: {{ fill: "F5F4F2", type: ShadingType.CLEAR }},
              margins: {{ top:80, bottom:80, left:120, right:120 }},
              children: [new Paragraph({{ children: [new TextRun({{ text: "Domain", bold: true, font: "Arial", size: 18 }})] }})] }}),
            new TableCell({{ borders: noBorders, width: {{ size: 2200, type: WidthType.DXA }},
              shading: {{ fill: "F5F4F2", type: ShadingType.CLEAR }},
              margins: {{ top:80, bottom:80, left:120, right:120 }},
              children: [new Paragraph({{ children: [new TextRun({{ text: {js(domain)}, font: "Arial", size: 18, color: "D93B1A" }})] }})] }}),
          ] }}),
          new TableRow({{ children: [
            new TableCell({{ borders: noBorders, width: {{ size: 1800, type: WidthType.DXA }},
              shading: {{ fill: "F5F4F2", type: ShadingType.CLEAR }},
              margins: {{ top:80, bottom:80, left:120, right:120 }},
              children: [new Paragraph({{ children: [new TextRun({{ text: "Audit date", bold: true, font: "Arial", size: 18 }})] }})] }}),
            new TableCell({{ borders: noBorders, width: {{ size: 2200, type: WidthType.DXA }},
              shading: {{ fill: "F5F4F2", type: ShadingType.CLEAR }},
              margins: {{ top:80, bottom:80, left:120, right:120 }},
              children: [new Paragraph({{ children: [new TextRun({{ text: {js(month_year)}, font: "Arial", size: 18 }})] }})] }}),
          ] }}),
          new TableRow({{ children: [
            new TableCell({{ borders: noBorders, width: {{ size: 1800, type: WidthType.DXA }},
              shading: {{ fill: "F5F4F2", type: ShadingType.CLEAR }},
              margins: {{ top:80, bottom:80, left:120, right:120 }},
              children: [new Paragraph({{ children: [new TextRun({{ text: "Prepared by", bold: true, font: "Arial", size: 18 }})] }})] }}),
            new TableCell({{ borders: noBorders, width: {{ size: 2200, type: WidthType.DXA }},
              shading: {{ fill: "F5F4F2", type: ShadingType.CLEAR }},
              margins: {{ top:80, bottom:80, left:120, right:120 }},
              children: [new Paragraph({{ children: [new TextRun({{ text: "Summit Media · AI Visibility Practice", font: "Arial", size: 18 }})] }})] }}),
          ] }}),
        ]
      }}),
      new Paragraph({{ spacing: {{ before: 160 }}, children: [
        new TextRun({{ text: "Confidential. Prepared for the " + {js(company)} + " digital team.", font: "Arial", size: 18, italics: true, color: "6B6B6B" }})
      ] }}),
      new Paragraph({{ children: [new PageBreak()] }}),

      // ── Executive Summary ──
      new Paragraph({{ heading: HeadingLevel.HEADING_1,
        children: [new TextRun({{ text: "EXECUTIVE SUMMARY", font: "Arial", size: 28, bold: true }})] }}),
      new Paragraph({{ spacing: {{ before: 80, after: 160 }},
        children: [new TextRun({{ text: {js(summary)}, font: "Arial", size: 20 }})] }}),

      // ── Headline scorecard ──
      new Paragraph({{ heading: HeadingLevel.HEADING_2,
        children: [new TextRun({{ text: "Headline Scorecard", font: "Arial", size: 24, bold: true }})] }}),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [3500, 900, 4960],
        rows: [{scorecard_rows_js}
        ]
      }}),

      // ── Dimension averages ──
      new Paragraph({{ heading: HeadingLevel.HEADING_2, spacing: {{ before: 320 }},
        children: [new TextRun({{ text: "Dimension Averages", font: "Arial", size: 24, bold: true }})] }}),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170],
        rows: [
          new TableRow({{ children: [{dim_avg_row1}
          ] }}),
        ]
      }}),
      new Paragraph({{ children: [new PageBreak()] }}),

      // ── Methodology ──
      new Paragraph({{ heading: HeadingLevel.HEADING_1,
        children: [new TextRun({{ text: "METHODOLOGY", font: "Arial", size: 28, bold: true }})] }}),
      new Paragraph({{ children: [new TextRun({{ text: "This audit treats {js(domain)[1:-1]}'s site the way ChatGPT, Perplexity, Gemini and Claude actually consume it. AI crawlers fetch a URL, parse the static HTML returned on first request, look for structured data, and decide whether the page is citable for a user's query. They do not execute a full JavaScript render in most cases.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ heading: HeadingLevel.HEADING_2,
        children: [new TextRun({{ text: "The Eight Dimensions", font: "Arial", size: 24, bold: true }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "ARIA implementation. Semantic landmarks, role attributes, descriptive aria-label values.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Structured data / schema markup. Schema.org JSON-LD — the single highest-leverage signal for AI citation.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Heading structure. A clean H1–H6 outline that gives crawlers a topical map of the page.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Meta and SEO signals. Title, description, canonical, Open Graph, Twitter Card, robots, language.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Link quality. Internal consistency, descriptive anchor text, protocol uniformity, link density.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Image alt text. Descriptive alt attributes — the cheapest accessibility-and-AI dual win available.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Crawlability and JS dependency. Whether content is present in static HTML or requires JS execution.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "LLM content signals. First-hand expertise, named authors, dates, citations, accreditations.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ heading: HeadingLevel.HEADING_2,
        children: [new TextRun({{ text: "Scoring Bands", font: "Arial", size: 24, bold: true }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Red (1–2): Critical. Actively blocking AI visibility. Fix first.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Amber (3–5): Capping. Under-performing relative to what's possible. High value to fix.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Green (6+): Working. Meets the bar AI crawlers expect.", font: "Arial", size: 20 }})] }}),
      new Paragraph({{ children: [new PageBreak()] }}),

      // ── Per-page sections ──
      {page_sections_js}

      // ── Cross-cutting themes ──
      new Paragraph({{ heading: HeadingLevel.HEADING_1,
        children: [new TextRun({{ text: "CROSS-CUTTING THEMES", font: "Arial", size: 28, bold: true }})] }}),
      {themes_js if themes_js else "new Paragraph({ children: [new TextRun({ text: 'No cross-cutting themes identified.', font: 'Arial', size: 20 })] }),"}
      new Paragraph({{ children: [new PageBreak()] }}),

      // ── Recommendations ──
      new Paragraph({{ heading: HeadingLevel.HEADING_1,
        children: [new TextRun({{ text: "PRIORITY RECOMMENDATIONS", font: "Arial", size: 28, bold: true }})] }}),
      new Paragraph({{ spacing: {{ after: 160 }}, children: [new TextRun({{ text: "Ranked by AI-citation impact relative to implementation cost. P1 = ship in the next sprint. P2 = ship this quarter. P3 = ship when convenient.", font: "Arial", size: 20, italics: true }})] }}),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [600, 4200, 1800, 1500, 1260],
        rows: [{rec_rows_js}
        ]
      }}),
    ]
  }}]
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync('/tmp/audit_output.docx', buf);
  console.log('OK');
}}).catch(e => {{ console.error(e); process.exit(1); }});
"""

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script)
        script_path = f.name

    result = subprocess.run(
        ["node", script_path],
        capture_output=True, text=True, timeout=60
    )
    os.unlink(script_path)

    if result.returncode != 0:
        raise RuntimeError(f"docx generation failed:\n{result.stderr}")

    with open("/tmp/audit_output.docx", "rb") as f:
        return f.read()


# ─── Build One-Pager PDF ──────────────────────────────────────────────────────
def build_onepager(data: dict, month_year: str) -> bytes:
    company = data.get("company_name", "Client")
    domain  = data.get("domain", "")
    avg     = data.get("average_score", 0)
    dim_avg = data.get("dimension_averages", {})
    summary_short = data.get("executive_summary", "")[:600]
    working = data.get("whats_working", [])[:3]
    holding = data.get("whats_holding_back", [])[:3]
    wins    = data.get("three_quick_wins", [])

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]

    SUMMIT_RED_RL   = colors.HexColor("#D93B1A")
    SUMMIT_DARK_RL  = colors.HexColor("#1A1A1A")
    SUMMIT_GREY_RL  = colors.HexColor("#6B6B6B")
    SUMMIT_LIGHT_RL = colors.HexColor("#F5F4F2")
    WHITE_RL        = colors.white

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    W = A4[0] - 30*mm

    def sty(name="Normal", **kw):
        return ParagraphStyle(name, fontName="Helvetica", fontSize=9,
                              leading=12, textColor=SUMMIT_DARK_RL, **kw)

    story = []

    # ── Header bar ──────────────────────────────────────────────────
    logo_w, logo_h = 35, 35
    header_data = [[
        RLImage(LOGO_PATH, width=logo_w, height=logo_h),
        Paragraph(
            f'<font name="Helvetica-Bold" size="7" color="#6B6B6B">'
            f'AI VISIBILITY SNAPSHOT<br/>{company.upper()} · {month_year.upper()}</font>',
            sty(alignment=TA_RIGHT)
        )
    ]]
    ht = Table(header_data, colWidths=[logo_w+5*mm, W - logo_w - 5*mm])
    ht.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW", (0,0), (-1,0), 2, SUMMIT_RED_RL),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(ht)
    story.append(Spacer(1, 6*mm))

    # ── Hero headline ────────────────────────────────────────────────
    story.append(Paragraph(
        f'<font name="Helvetica-Bold" size="22">Is your site ready<br/>'
        f'for the <font color="#D93B1A"><i>AI search era?</i></font></font>',
        sty(alignment=TA_LEFT, leading=28)
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f'We audited <b>{domain}</b> the way ChatGPT, Perplexity, Gemini and Claude see it. '
        + summary_short.replace("\n"," "),
        sty(fontSize=8.5, leading=12)
    ))
    story.append(Spacer(1, 4*mm))

    # ── Score block ──────────────────────────────────────────────────
    score_col = W * 0.22
    text_col  = W - score_col - 4*mm
    score_data = [[
        Paragraph(
            f'<font name="Helvetica-Bold" size="52" color="#D93B1A">{avg}</font>'
            f'<font name="Helvetica" size="20" color="#6B6B6B">/80</font>',
            sty(alignment=TA_CENTER)
        ),
        Paragraph(
            '<font name="Helvetica-Bold" size="11">AVERAGE PAGE SCORE<br/></font>'
            '<font name="Helvetica-Bold" size="14">A solid foundation. A clear AI gap.</font>',
            sty(leading=18)
        )
    ]]
    st_ = Table(score_data, colWidths=[score_col, text_col])
    st_.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,-1), SUMMIT_LIGHT_RL),
        ("ROUNDEDCORNERS", [3]),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(st_)
    story.append(Spacer(1, 4*mm))

    # ── Dimension scores row ─────────────────────────────────────────
    cell_w = W / 8
    dim_row = []
    for dk, dl in zip(dim_keys, dim_labels):
        s = dim_avg.get(dk, 0)
        c = colors.HexColor(score_color(s))
        dim_row.append(
            Paragraph(
                f'<font name="Helvetica" size="6" color="#FFFFFF">{dl}<br/></font>'
                f'<font name="Helvetica-Bold" size="14" color="#FFFFFF">{s}</font>'
                f'<font name="Helvetica" size="8" color="#FFFFFF">/10</font>',
                sty(alignment=TA_CENTER, leading=11)
            )
        )
    dim_t = Table([dim_row], colWidths=[cell_w]*8)
    dim_styles = [("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]
    for i, dk in enumerate(dim_keys):
        s = dim_avg.get(dk, 0)
        c = colors.HexColor(score_color(s))
        dim_styles.append(("BACKGROUND",(i,0),(i,0), c))
    dim_t.setStyle(TableStyle(dim_styles))
    story.append(dim_t)
    story.append(Spacer(1, 5*mm))

    # ── What's working / holding back ───────────────────────────────
    half = (W - 6*mm) / 2

    def bullet_rows(items, icon, icon_color):
        rows = []
        for it in items:
            pt  = it.get("point","")
            det = it.get("detail","")
            rows.append([
                Paragraph(f'<font color="{icon_color}" name="Helvetica-Bold" size="11">{icon}</font>',
                          sty(alignment=TA_CENTER)),
                Paragraph(f'<b>{pt}</b><br/><font size="7.5">{det}</font>', sty(fontSize=8, leading=11))
            ])
        return rows

    working_rows = bullet_rows(working, "✚", "#27AE60")
    holding_rows = bullet_rows(holding, "!", "#D93B1A")

    def side_table(rows, header):
        data = [[Paragraph(f'<font name="Helvetica-Bold" size="9" color="#FFFFFF">{header}</font>',
                           sty(alignment=TA_LEFT))]] + rows
        t = Table(data, colWidths=[8*mm, half - 8*mm])
        bg = "#27AE60" if "WORKING" in header else "#D93B1A"
        ts = [
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor(bg)),
            ("SPAN",(0,0),(-1,0)),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),5),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LINEBELOW",(0,0),(-1,-1),0.3, colors.HexColor("#DDDDDD")),
        ]
        t.setStyle(TableStyle(ts))
        return t

    two_col = Table([[side_table(working_rows,"✚ WHAT'S WORKING"),
                      side_table(holding_rows,"! WHAT'S HOLDING YOU BACK")]],
                    colWidths=[half, half], hAlign="LEFT")
    two_col.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                                  ("RIGHTPADDING",(0,0),(-1,-1),6),]))
    story.append(two_col)
    story.append(Spacer(1, 5*mm))

    # ── Three quick wins ─────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=SUMMIT_LIGHT_RL))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '<font name="Helvetica" size="7" color="#6B6B6B">THREE MOVES THAT MOVE THE NEEDLE</font><br/>'
        '<font name="Helvetica-Bold" size="14">Quick wins, big impact</font>',
        sty(leading=18)
    ))
    story.append(Spacer(1, 3*mm))

    third = W / 3
    win_data = []
    win_cells = []
    for w in wins[:3]:
        win_cells.append(
            Paragraph(
                f'<font name="Helvetica-Bold" size="28" color="#D93B1A">{w.get("number","")}</font><br/>'
                f'<font name="Helvetica-Bold" size="9">{w.get("title","")}</font><br/>'
                f'<font name="Helvetica" size="8">{w.get("detail","")}</font>',
                sty(leading=12)
            )
        )
    if win_cells:
        wins_t = Table([win_cells], colWidths=[third]*3)
        wins_t.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),0),
            ("LEFTPADDING",(1,0),(-1,-1),8),
            ("LINEAFTER",(0,0),(1,-1),0.5,SUMMIT_LIGHT_RL),
        ]))
        story.append(wins_t)

    story.append(Spacer(1, 5*mm))

    # ── CTA footer ───────────────────────────────────────────────────
    cta_data = [[
        Paragraph(
            f'<font name="Helvetica-Bold" size="10">We\'ll walk your team through<br/>'
            f'every finding. <font color="#D93B1A"><i>No obligation.</i></font></font>',
            sty()
        ),
        Paragraph(
            '<font name="Helvetica" size="7" color="#6B6B6B">BOOK A SESSION<br/></font>'
            '<font name="Helvetica-Bold" size="11">hello@summitmedia.com</font>',
            sty(alignment=TA_CENTER)
        )
    ]]
    cta_t = Table(cta_data, colWidths=[W*0.55, W*0.45])
    cta_t.setStyle(TableStyle([
        ("BACKGROUND",(1,0),(1,0), SUMMIT_LIGHT_RL),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEABOVE",(0,0),(-1,0),2,SUMMIT_RED_RL),
    ]))
    story.append(cta_t)

    # ── Footer bar ───────────────────────────────────────────────────
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<font name="Helvetica" size="7" color="#6B6B6B">'
        'SUMMITMEDIA.CO.UK &nbsp;&nbsp;&nbsp; PREPARED BY SUMMIT · AI VISIBILITY PRACTICE'
        '</font>',
        sty(alignment=TA_CENTER)
    ))

    doc.build(story)
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
</style>
<div class="summit-header">
  <h1>🔍 Summit · AI Visibility Audit Tool</h1>
  <p>Audit any website the way ChatGPT, Perplexity, Gemini and Claude see it.</p>
</div>
""", unsafe_allow_html=True)

# ─── Inputs ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    domain = st.text_input("Domain to audit", placeholder="e.g. healix.com")
with col2:
    month_year = st.text_input("Audit month / year", value=datetime.now().strftime("%B %Y"))

st.markdown("**Additional pages to audit** (up to 3 internal URLs, one per line)")
extra_raw = st.text_area("", placeholder="https://healix.com/about\nhttps://healix.com/services", height=80)
extra_urls = [u for u in extra_raw.strip().splitlines() if u.strip()]

run = st.button("🚀 Run Audit", use_container_width=True)

if run:
    if not domain:
        st.error("Please enter a domain.")
        st.stop()

    model = get_gemini_client()

    with st.spinner("Fetching pages…"):
        pages = fetch_pages(domain, extra_urls)

    st.success(f"Fetched {len(pages)} page(s). Running Gemini audit…")

    with st.spinner("Analysing with Gemini…"):
        try:
            audit = run_audit(model, pages)
        except Exception as e:
            st.error(f"Gemini audit failed: {e}")
            st.stop()

    # ── Display results ────────────────────────────────────────────────────
    company = audit.get("company_name", domain)
    avg     = audit.get("average_score", 0)
    dim_avg = audit.get("dimension_averages", {})
    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]

    st.markdown(f"## 📊 Results: {company}")

    # Score overview
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"""
        <div class="score-card">
          <div style="font-size:0.7rem;font-weight:600;color:{SUMMIT_GREY};letter-spacing:1px">AVERAGE PAGE SCORE</div>
          <div class="score-big">{avg}</div>
          <div style="color:{SUMMIT_GREY}">/80</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        badges = ""
        for dk, dl in zip(dim_keys, dim_labels):
            s = dim_avg.get(dk, 0)
            badges += f'<span class="dim-badge" style="background:{score_color(s)}">{dl}: {s}/10</span>'
        st.markdown(f'<div style="padding:1rem">{badges}</div>', unsafe_allow_html=True)
        st.markdown(f"**Executive summary:** {audit.get('executive_summary','')[:400]}…")

    # Page tabs
    pages_data = audit.get("pages", [])
    if pages_data:
        tabs = st.tabs([p.get("title", f"Page {i+1}") for i, p in enumerate(pages_data)])
        for tab, page in zip(tabs, pages_data):
            with tab:
                st.markdown(f"**Score:** {page.get('score',0)}/80 &nbsp;|&nbsp; *{page.get('verdict','')}*")
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

    # ── Generate documents ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Outputs")
    col_d, col_p = st.columns(2)

    with col_d:
        with st.spinner("Building Word document…"):
            try:
                docx_bytes = build_docx(audit, month_year)
                st.download_button(
                    "⬇️ Download Full Audit (.docx)",
                    data=docx_bytes,
                    file_name=f"summit-ai-audit-{company.lower().replace(' ','-')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Word doc error: {e}")

    with col_p:
        with st.spinner("Building one-pager PDF…"):
            try:
                pdf_bytes = build_onepager(audit, month_year)
                st.download_button(
                    "⬇️ Download One-Pager (.pdf)",
                    data=pdf_bytes,
                    file_name=f"summit-ai-snapshot-{company.lower().replace(' ','-')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF error: {e}")
