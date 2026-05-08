# Summit · AI Visibility Audit Tool

A Streamlit app that audits any website the way ChatGPT, Perplexity, Gemini and Claude see it — and outputs a branded Word doc + one-pager PDF.

## Setup

### 1. Files needed in your repo root
```
app.py
requirements.txt
summit_logo.png        ← copy the Summit logo PNG here
```

### 2. Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file to `app.py`

### 3. Add your Gemini API key
In Streamlit Cloud → your app → **Settings → Secrets**, paste:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```
Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Add Node.js support
Streamlit Cloud supports Node.js. Add a `packages.txt` file:
```
nodejs
npm
```
Then in a `setup.sh` or via `packages.txt`, ensure `docx` npm package is installed.

For Streamlit Cloud, add a `packages.txt`:
```
nodejs
npm
```
And add a `.streamlit/config.toml`:
```toml
[server]
runOnSave = false
```

The app will auto-run `npm install -g docx` on first use if not present.

## Usage

1. Enter the domain to audit (e.g. `healix.com`)
2. Optionally add up to 3 specific internal URLs to audit
3. Set the audit month/year label
4. Click **Run Audit**
5. Download the `.docx` full report and `.pdf` one-pager

## Output

| Output | Description |
|--------|-------------|
| **Full Audit (.docx)** | Multi-page Word doc matching the Inspire Sport template — cover page, executive summary, scorecard, methodology, per-page breakdowns, cross-cutting themes, priority recommendations |
| **One-Pager (.pdf)** | Single A4 PDF matching the Healix snapshot — score, dimension badges, what's working/holding back, three quick wins, CTA |

Both outputs are branded with Summit's red (#D93B1A) colour scheme and Helvetica/Arial fonts.

## Scoring dimensions

Each page is scored 1–10 across 8 dimensions (max 80):

| Dimension | What it measures |
|-----------|-----------------|
| ARIA | Semantic landmarks, role attributes, aria-labels |
| SCHEMA | schema.org JSON-LD structured data |
| HEADINGS | H1–H6 hierarchy and topic signal |
| META | Title, OG, Twitter Card, canonical |
| LINKS | Internal link quality, anchor text, protocols |
| ALT TEXT | Image alt attribute quality |
| CRAWL | Static HTML vs JS dependency |
| LLM | First-hand expertise, named entities, authority signals |

**Score bands:** 🔴 1–2 Critical · 🟡 3–5 Capping · 🟢 6+ Working
