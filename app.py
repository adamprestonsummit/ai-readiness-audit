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
    return genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

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
    """
    Writes audit data to a JSON file, then runs a static Node.js script
    that reads the JSON. This avoids injecting any user/LLM content into
    JS source code, which was causing SyntaxErrors.
    """
    import tempfile, os, subprocess, json as _json

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]

    def score_color_hex(s):
        if s <= 2: return "C0392B"
        if s <= 5: return "E67E22"
        return "27AE60"

    # Enrich data with derived fields the JS needs
    enriched = _json.loads(_json.dumps(data))   # deep copy via json
    enriched["month_year"]   = month_year
    enriched["logo_path"]    = LOGO_PATH
    enriched["dim_keys"]     = dim_keys
    enriched["dim_labels"]   = dim_labels
    # Add colour fields
    dim_avg = enriched.get("dimension_averages", {})
    enriched["dim_colors"] = {k: score_color_hex(dim_avg.get(k, 0)) for k in dim_keys}
    for page in enriched.get("pages", []):
        dims = page.get("dimensions", {})
        page["dim_colors"] = {k: score_color_hex(dims.get(k, {}).get("score", 0)) for k in dim_keys}
        page["score_color"] = score_color_hex(page.get("score", 0))

    # Ensure docx npm package is available — install locally if not found globally
    app_dir = os.path.dirname(os.path.abspath(__file__))
    node_modules = os.path.join(app_dir, "node_modules")
    if not os.path.exists(os.path.join(node_modules, "docx")):
        subprocess.run(
            ["npm", "install", "docx", "--prefix", app_dir],
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
const headerChildren = [
  new Table({
    width: { size: 9906, type: WidthType.DXA },
    columnWidths: [1200, 8706],
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
    txt('\t\tPREPARED BY SUMMIT \u00b7 AI VISIBILITY PRACTICE', { size: 16, color: '6B6B6B' }),
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
    cell([para([txt((p.score||0)+'/80', { bold: true, size: 18 })], { alignment: AlignmentType.CENTER })], 900, shade),
    cell([para([txt(p.verdict || '', { size: 18 })])], 4960, shade),
  ]}));
});

// ── Dimension averages row ───────────────────────────────────────────────────
const dimAvgCells = dimKeys.map(function(dk, i) {
  const s = dimAvg[dk] || 0;
  const col = dimColors[dk] || '27AE60';
  return cell([
    para([txt(dimLabels[i], { size: 14, bold: true, color: 'FFFFFF' })], { alignment: AlignmentType.CENTER }),
    para([txt(s+'/10', { size: 24, bold: true, color: 'FFFFFF' })], { alignment: AlignmentType.CENTER }),
  ], 1170, { shading: { fill: col, type: ShadingType.CLEAR } });
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
    para([txt('Total score: '+(p.score||0)+'/80', { bold: true, size: 20 })]),
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
            cell([para([txt('Summit Media \u00b7 AI Visibility Practice', { size: 18 })])], 2200,
              { borders: noBorders, shading: { fill: 'F5F4F2', type: ShadingType.CLEAR } }),
          ]}),
        ]
      }),
      para([txt('Confidential. Prepared for the ' + company + ' digital team.', { size: 18, italics: true, color: '6B6B6B' })],
        { spacing: { before: 160 } }),
      para([new PageBreak()]),

      // Executive Summary
      para([txt('EXECUTIVE SUMMARY', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      para([txt(summary, { size: 20 })], { spacing: { before: 80, after: 160 } }),

      // Scorecard
      para([txt('Headline Scorecard', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3500, 900, 4960], rows: scorecardRows }),

      // Dimension averages
      para([txt('Dimension Averages', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2, spacing: { before: 320 } }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170],
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
    global_modules = subprocess.run(
        ["npm", "root", "-g"], capture_output=True, text=True
    ).stdout.strip()
    env["NODE_PATH"] = local_modules + os.pathsep + global_modules
    result = subprocess.run(
        ["node", script_file.name, data_file.name, out_path],
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
    import re as _re

    company       = data.get("company_name", "Client")
    domain        = data.get("domain", "")
    avg           = int(data.get("average_score", 0))
    dim_avg       = data.get("dimension_averages", {})
    exec_summary  = data.get("executive_summary", "")
    working       = data.get("whats_working", [])[:3]
    holding       = data.get("whats_holding_back", [])[:3]
    wins          = data.get("three_quick_wins", [])[:3]

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]

    RED   = colors.HexColor("#D93B1A")
    DARK  = colors.HexColor("#1A1A1A")
    GREY  = colors.HexColor("#6B6B6B")
    LIGHT = colors.HexColor("#F5F4F2")
    GREEN = colors.HexColor("#27AE60")
    AMBER = colors.HexColor("#E67E22")
    WHITE = colors.white

    def dim_color(s):
        if s <= 2: return colors.HexColor("#C0392B")
        if s <= 5: return colors.HexColor("#E67E22")
        return colors.HexColor("#27AE60")

    def safe(text):
        """Escape XML special chars for ReportLab paragraphs."""
        return str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    buf = BytesIO()

    # Page: A4, tight margins to give maximum layout room
    PAGE_W, PAGE_H = A4
    ML = MR = 14*mm
    MT = MB = 12*mm
    W = PAGE_W - ML - MR   # usable width ~182mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
    )

    # ── Style factory ────────────────────────────────────────────────
    _style_cache = {}
    def S(name, **kw):
        key = name + str(sorted(kw.items()))
        if key not in _style_cache:
            base = dict(fontName="Helvetica", fontSize=8, leading=11,
                        textColor=DARK, spaceAfter=0, spaceBefore=0)
            base.update(kw)
            _style_cache[key] = ParagraphStyle(name + str(len(_style_cache)), **base)
        return _style_cache[key]

    def P(markup, **kw):
        return Paragraph(markup, S("p", **kw))

    story = []

    # ════════════════════════════════════════════════════════════════
    # 1. HEADER — logo left, tag right
    # ════════════════════════════════════════════════════════════════
    LOGO_W = LOGO_H = 32

    logo_cell = RLImage(LOGO_PATH, width=LOGO_W, height=LOGO_H) if os.path.exists(LOGO_PATH) else P("")
    tag_markup = (
        f'<font name="Helvetica" size="6.5" color="#6B6B6B">'
        f'AI VISIBILITY SNAPSHOT<br/>'
        f'{safe(company).upper()} &middot; {safe(month_year).upper()}</font>'
    )
    hdr = Table(
        [[logo_cell, P(tag_markup, alignment=TA_RIGHT, leading=9)]],
        colWidths=[LOGO_W + 4*mm, W - LOGO_W - 4*mm],
        rowHeights=[LOGO_H],
    )
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LINEBELOW",     (0,0), (-1,-1), 1.5, RED),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 5*mm))

    # ════════════════════════════════════════════════════════════════
    # 2. HERO HEADLINE + INTRO
    # ════════════════════════════════════════════════════════════════
    story.append(P(
        '<font name="Helvetica-Bold" size="20">Is your site ready<br/>'
        f'for the <font color="#D93B1A"><i>AI search era?</i></font></font>',
        leading=26,
    ))
    story.append(Spacer(1, 3*mm))

    # Intro: domain bold + first ~280 chars of summary
    intro_text = safe(exec_summary)[:280].strip()
    if len(exec_summary) > 280:
        intro_text += "…"
    story.append(P(
        f'We audited <b>{safe(domain)}</b> the way ChatGPT, Perplexity, Gemini and Claude see it. '
        + intro_text,
        fontSize=8, leading=11,
    ))
    story.append(Spacer(1, 4*mm))

    # ════════════════════════════════════════════════════════════════
    # 3. SCORE BOX  (score left | label + tagline right)
    # ════════════════════════════════════════════════════════════════
    SCORE_COL = 38*mm
    TEXT_COL  = W - SCORE_COL

    score_markup = (
        f'<font name="Helvetica-Bold" size="48" color="#D93B1A">{avg}</font>'
        f'<font name="Helvetica" size="16" color="#6B6B6B">/80</font>'
    )
    label_markup = (
        '<font name="Helvetica" size="7" color="#6B6B6B">AVERAGE PAGE SCORE</font><br/>'
        '<font name="Helvetica-Bold" size="13" color="#1A1A1A">A solid foundation.</font><br/>'
        '<font name="Helvetica-Bold" size="13" color="#D93B1A">A clear AI gap.</font>'
    )

    score_box = Table(
        [[P(score_markup, alignment=TA_CENTER, leading=52),
          P(label_markup, leading=17)]],
        colWidths=[SCORE_COL, TEXT_COL],
    )
    score_box.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (0,0),   8),
        ("LEFTPADDING",   (1,0), (1,0),   10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(score_box)
    story.append(Spacer(1, 3*mm))

    # ════════════════════════════════════════════════════════════════
    # 4. DIMENSION BADGES ROW
    # ════════════════════════════════════════════════════════════════
    cell_w = W / 8
    dim_cells = []
    for dk, dl in zip(dim_keys, dim_labels):
        s = dim_avg.get(dk, 0)
        dim_cells.append(P(
            f'<font name="Helvetica" size="5.5" color="#FFFFFF">{dl}<br/></font>'
            f'<font name="Helvetica-Bold" size="13" color="#FFFFFF">{s}</font>'
            f'<font name="Helvetica" size="7" color="#FFFFFF">/10</font>',
            alignment=TA_CENTER, leading=10,
        ))

    dim_t = Table([dim_cells], colWidths=[cell_w]*8)
    dim_styles = [
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 1),
        ("RIGHTPADDING",  (0,0), (-1,-1), 1),
    ]
    for i, dk in enumerate(dim_keys):
        dim_styles.append(("BACKGROUND", (i,0), (i,0), dim_color(dim_avg.get(dk,0))))
    dim_t.setStyle(TableStyle(dim_styles))
    story.append(dim_t)
    story.append(Spacer(1, 5*mm))

    # ════════════════════════════════════════════════════════════════
    # 5. WHAT'S WORKING / HOLDING YOU BACK
    #    Two independent tables placed side-by-side via a wrapper table
    #    Key fix: build each side as a list of Paragraphs, NOT nested Tables
    # ════════════════════════════════════════════════════════════════
    GAP   = 5*mm
    HALF  = (W - GAP) / 2

    def build_side(items, hdr_text, hdr_color, icon, icon_color_hex):
        """Returns a Table for one side (working or holding)."""
        rows = []
        # Header
        rows.append([
            P(f'<font name="Helvetica-Bold" size="8.5" color="#FFFFFF">{hdr_text}</font>',
              leading=11),
        ])
        # Up to 3 bullet rows
        for it in items[:3]:
            pt  = safe(it.get("point",""))
            det = safe(it.get("detail",""))
            rows.append([
                P(
                    f'<font name="Helvetica-Bold" size="8" color="{icon_color_hex}">{icon} </font>'
                    f'<font name="Helvetica-Bold" size="8">{pt}</font><br/>'
                    f'<font name="Helvetica" size="7" color="#4A4A4A">{det}</font>',
                    leading=11,
                ),
            ])
        # Pad to 3 rows so both sides same height
        while len(rows) < 4:
            rows.append([P(" ", fontSize=7)])

        t = Table(rows, colWidths=[HALF])
        ts = [
            # Header row bg
            ("BACKGROUND",    (0,0), (-1,0),  hdr_color),
            ("TOPPADDING",    (0,0), (-1,-1),  5),
            ("BOTTOMPADDING", (0,0), (-1,-1),  5),
            ("LEFTPADDING",   (0,0), (-1,-1),  7),
            ("RIGHTPADDING",  (0,0), (-1,-1),  7),
            ("VALIGN",        (0,0), (-1,-1),  "TOP"),
            # Light bg on content rows
            ("BACKGROUND",    (0,1), (-1,-1),  colors.HexColor("#F9F9F9")),
            # Divider lines between content rows
            ("LINEBELOW",     (0,1), (-1,-2),  0.4, colors.HexColor("#E0E0E0")),
        ]
        t.setStyle(TableStyle(ts))
        return t

    working_t = build_side(working, "\u271a WHAT\u2019S WORKING",    GREEN, "\u271a", "#27AE60")
    holding_t = build_side(holding, "! WHAT\u2019S HOLDING YOU BACK", RED,   "!",     "#D93B1A")

    sides = Table(
        [[working_t, Spacer(GAP, 1), holding_t]],
        colWidths=[HALF, GAP, HALF],
    )
    sides.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(sides)
    story.append(Spacer(1, 6*mm))

    # ════════════════════════════════════════════════════════════════
    # 6. THREE QUICK WINS
    # ════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#DDDDDD"), spaceAfter=4))
    story.append(P(
        '<font name="Helvetica" size="6.5" color="#6B6B6B">THREE MOVES THAT MOVE THE NEEDLE</font>',
        leading=9,
    ))
    story.append(P(
        '<font name="Helvetica-Bold" size="13">Quick wins, big impact</font>',
        leading=17,
    ))
    story.append(Spacer(1, 4*mm))

    THIRD = W / 3
    win_rows = []
    for w in wins:
        num    = safe(w.get("number",""))
        title  = safe(w.get("title",""))
        detail = safe(w.get("detail",""))
        win_rows.append(
            P(
                f'<font name="Helvetica-Bold" size="26" color="#D93B1A">{num}</font><br/>'
                f'<font name="Helvetica-Bold" size="8.5">{title}</font><br/>'
                f'<font name="Helvetica" size="7.5" color="#4A4A4A">{detail}</font>',
                leading=13,
            )
        )
    # Pad to 3
    while len(win_rows) < 3:
        win_rows.append(P(" "))

    wins_t = Table([win_rows], colWidths=[THIRD]*3)
    wins_t.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING",   (0,0), (0,0),   0),
        ("LEFTPADDING",   (1,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("LINEAFTER",     (0,0), (1,-1),  0.5, colors.HexColor("#DDDDDD")),
    ]))
    story.append(wins_t)
    story.append(Spacer(1, 6*mm))

    # ════════════════════════════════════════════════════════════════
    # 7. CTA BAR
    # ════════════════════════════════════════════════════════════════
    cta_left = P(
        '<font name="Helvetica-Bold" size="10">We\u2019ll walk your team through<br/>'
        'every finding. <font color="#D93B1A"><i>No obligation.</i></font></font>',
        leading=14,
    )
    cta_right = P(
        '<font name="Helvetica" size="6.5" color="#6B6B6B">BOOK A SESSION<br/></font>'
        '<font name="Helvetica-Bold" size="10.5">hello@summitmedia.com</font>',
        alignment=TA_CENTER, leading=13,
    )
    cta_t = Table([[cta_left, cta_right]], colWidths=[W*0.52, W*0.48])
    cta_t.setStyle(TableStyle([
        ("BACKGROUND",    (1,0), (1,0),  LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("LEFTPADDING",   (0,0), (0,0),   10),
        ("LEFTPADDING",   (1,0), (1,0),   8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LINEABOVE",     (0,0), (-1,0),  2, RED),
    ]))
    story.append(cta_t)
    story.append(Spacer(1, 3*mm))

    # ════════════════════════════════════════════════════════════════
    # 8. FOOTER TEXT
    # ════════════════════════════════════════════════════════════════
    story.append(P(
        '<font name="Helvetica" size="6.5" color="#6B6B6B">'
        'SUMMITMEDIA.CO.UK &nbsp;&nbsp;&nbsp;&nbsp; '
        'PREPARED BY SUMMIT &middot; AI VISIBILITY PRACTICE'
        '</font>',
        alignment=TA_CENTER, leading=9,
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
