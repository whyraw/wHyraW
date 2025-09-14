# Main.py
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3, io, os, re
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
import numpy as np
from dotenv import load_dotenv


DB_PATH = "user_data.db"
TABLE_NAME = "data"

load_dotenv()
#helpers for charts --  start

CHART_KEYWORDS = {
    "bar": ["bar", "bar chart", "histogram"],
    "line": ["line", "trend", "time series", "timeline"],
    "pie": ["pie", "donut", "doughnut", "circle"],
    "stacked": ["stacked", "grouped"],
}

def detect_chart_type(q: str) -> str:
    q = (q or "").lower()
    for t, kws in CHART_KEYWORDS.items():
        if any(k in q for k in kws):
            return t
    return "bar"  # default

def wants_chart(q: str, want_flag: Optional[bool] = None) -> bool:
    if want_flag is True:
        return True
    q = (q or "").lower()
    return any(k in q for k in CHART_KEYWORDS)

def is_date_like(series: pd.Series) -> bool:
    if np.issubdtype(series.dtype, np.datetime64):
        return True
    # strings that look like dates
    if series.dtype == object:
        try:
            pd.to_datetime(series.dropna().head(5), errors="raise")
            return True
        except Exception:
            return False
    return False

def build_chart_from_df(df: pd.DataFrame, forced_type: str = None) -> Optional[Dict[str, Any]]:
    """
    Infer a chart spec from a DataFrame, but allow user to force type via forced_type.
    Returns a Chart.js config dict.
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    if len(cols) < 2:
        return None

    # ---------------- Inference heuristics ----------------
    c0, c1 = cols[0], cols[1]
    value_col = cols[-1]
    maybe_date_x = is_date_like(df[c0])

    chart = None

    # --- 2 columns ---
    if len(cols) == 2:
        vals = pd.to_numeric(df[c1], errors="coerce").fillna(0)
        if maybe_date_x:
            # time series → line
            labels = pd.to_datetime(df[c0], errors="coerce").dt.strftime("%Y-%m-%d").fillna(df[c0].astype(str))
            chart = {
                "type": "line",
                "data": {
                    "labels": labels.tolist(),
                    "datasets": [{"label": str(c1), "data": vals.tolist()}]
                },
                "options": {"responsive": True, "plugins": {"legend": {"display": True}}}
            }
        else:
            # categorical → bar
            labels = df[c0].astype(str).tolist()
            chart = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": str(c1), "data": vals.tolist()}]
                },
                "options": {"responsive": True, "plugins": {"legend": {"display": True}}}
            }

    # --- 3+ columns ---
    else:
        x_col, series_col, value_col = cols[0], cols[1], cols[-1]
        x_vals = df[x_col].astype(str).tolist()
        value_vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

        series_names = df[series_col].astype(str).unique().tolist()
        x_order = list(dict.fromkeys(x_vals))

        datasets = []
        for s in series_names:
            sub = df[df[series_col].astype(str) == s]
            data_map = {str(k): float(v) for k, v in zip(sub[x_col].astype(str), pd.to_numeric(sub[value_col], errors="coerce").fillna(0))}
            datasets.append({
                "label": s,
                "data": [data_map.get(x, 0.0) for x in x_order]
            })

        chart_type = "line" if is_date_like(df[x_col]) else "bar"
        chart = {
            "type": chart_type,
            "data": {"labels": x_order, "datasets": datasets},
            "options": {
                "responsive": True,
                "plugins": {"legend": {"display": True}},
                "scales": {"x": {"stacked": (chart_type == "bar")},
                           "y": {"stacked": (chart_type == "bar")}}
            }
        }

    # ---------------- Forced override ----------------
    if forced_type:
        chart["type"] = forced_type

        # Special handling for pie/doughnut → only 2 columns needed
        if forced_type in ["pie", "doughnut"]:
            if df.shape[1] >= 2:
                labels = df.iloc[:, 0].astype(str).tolist()
                values = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).tolist()
                chart = {
                    "type": forced_type,
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": "count", "data": values}]
                    },
                    "options": {"responsive": True, "plugins": {"legend": {"display": True}}}
                }

    return chart



#helpers for charts -- end

#helpers for column name fuzzy matching -- start
from rapidfuzz import process, fuzz

def get_table_columns(db_path: str, table: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]

def get_sample_values(db_path: str, table: str, col: str, n: int = 5) -> list[str]:
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(f'SELECT "{col}" FROM {table} WHERE "{col}" IS NOT NULL LIMIT {n}')
            return [str(r[0]) for r in cur.fetchall()]
    except Exception:
        return []

def normalize_term(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def build_alias_candidates(columns: list[str]) -> dict[str, str]:
    """Return dict[alias] = canonical_column for simple variations."""
    mapping: dict[str, str] = {}
    for c in columns:
        base = normalize_term(c)
        alts = {base, base.replace("_", " "), base.replace(" ", "_")}
        if base.endswith("s"): 
            alts.add(base[:-1])
        else: 
            alts.add(base + "s")
        for a in alts:
            if a:
                mapping[a] = c
    return mapping

def fuzzy_map_terms_to_columns(user_text: str, columns: list[str], cutoff: int = 85) -> dict[str, str]:
    """
    Extract tokens from user_text and fuzzy match to known columns.
    Safe against no-match (returns empty dict).
    """
    if not user_text or not columns:
        return {}

    tokens = [normalize_term(t) for t in re.findall(r"[A-Za-z0-9_]+", user_text)]
    tokens = [t for t in tokens if t]
    bigrams = [" ".join([tokens[i], tokens[i+1]]) for i in range(len(tokens)-1)] if len(tokens) > 1 else []

    candidates = build_alias_candidates(columns)
    keys = list(candidates.keys())
    mapping: dict[str, str] = {}

    # Try bigrams first, then unigrams
    for term in bigrams + tokens:
        if len(term) < 2:
            continue
        res = process.extractOne(term, keys, scorer=fuzz.WRatio, score_cutoff=cutoff)
        if res is None:
            continue
        match, score, _ = res  # safe now
        mapping[term] = candidates[match]
    return mapping
# helpers ended
# -------------------- FastAPI app & CORS --------------------
app = FastAPI(title="WhyRaw API", version="0.1.0")
@app.get("/")
def root():
    return {"status": "ok", "service": "whyraw-api"}

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",   # if you use a local file server
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://localhost:3000",
    "https://whyraw.ai",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"],  # keep * during MVP; tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Azure OpenAI client --------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "whyraw-gpt")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION
    )

# -------------------- Helpers --------------------
def clean_sql(text: str) -> str:
    """Strip ```sql fences / trailing semicolons and keep first statement."""
    if not text:
        return ""
    # remove markdown code fences
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M)
    text = text.strip()
    # keep first statement only
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts[0] if parts else ""

def try_read_csv_bytes(b: bytes) -> pd.DataFrame:
    """Try multiple encodings & auto-delimiter CSV parsing."""
    # quick tries
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc)
        except Exception:
            pass
    # tolerant encodings
    for enc in ("cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc, engine="python", sep=None)
        except Exception:
            pass
    # last-resort: replace undecodable chars
    return pd.read_csv(io.BytesIO(b), encoding="latin1", errors="replace", engine="python", sep=None)

def maybe_read_excel(b: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_excel(io.BytesIO(b))  # requires openpyxl
    except Exception:
        return None

def ensure_table_exists():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"SELECT 1 FROM {TABLE_NAME} LIMIT 1")
    except Exception:
        raise HTTPException(status_code=400, detail="No data. Upload a CSV/Excel to /upload first.")

# -------------------- Schemas --------------------
class AskBody(BaseModel):
    question: str
    want_chart: Optional[bool] = None


# -------------------- Endpoints --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "azure_openai_configured": bool(client),
        "endpoint": AZURE_OPENAI_ENDPOINT or None,
        "deployment": AZURE_DEPLOYMENT or None
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()

    # Excel if extension suggests
    df = None
    if file.filename and re.search(r"\.(xlsx?|xls)$", file.filename, re.I):
        df = maybe_read_excel(raw)

    if df is None:
        try:
            df = try_read_csv_bytes(raw)
        except Exception as e:
            return {
                "error": f"Could not parse file as CSV/Excel: {e}",
                "hint": "Try saving as UTF-8 CSV or upload the original Excel (.xlsx)."
            }

    # Basic cleanup
    df.columns = [str(c).strip() for c in df.columns]

    # try parse columns named 'date' (case-insensitive)
    for c in list(df.columns):
        if c.lower() == "date" or c.lower().endswith("_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    return {
        "message": "uploaded",
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "note": "Parsed with robust encoding/delimiter/Excel handling."
    }

@app.post("/ask")
@app.post("/ask")
async def ask(body: AskBody):
    if not client:
        return {"error": "Azure OpenAI is not configured. Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT."}

    ensure_table_exists()

    # --- Schema & fuzzy mapping ---
    columns = get_table_columns(DB_PATH, TABLE_NAME)
    if not columns:
        return {"error": "No columns found in table. Upload a dataset again."}

    samples = {c: get_sample_values(DB_PATH, TABLE_NAME, c, n=3) for c in columns}
    term_to_col = fuzzy_map_terms_to_columns(body.question, columns)

    # --- Prompting ---
    system = (
        "You are a SQL generator. "
        f"Output ONLY valid SQLite SQL for a table named `{TABLE_NAME}`. "
        "No explanations, no markdown, no comments. Return SQL only.\n"
        "Rules:\n"
        "1) Use ONLY these columns exactly as named.\n"
        "2) If the user refers to a synonym, map it to the closest column from the provided schema/mappings.\n"
        "3) If the user asks for a pivot across categories, return a TALL result with three columns (dimension, category, value). Do NOT hardcode values.\n"
        "4) Use double quotes around column names if needed.\n"
        "5) Keep queries simple SELECTs with WHERE, GROUP BY, ORDER BY, LIMIT."
    )

    schema_note = {
        "columns": columns,
        "example_values": samples,
        "fuzzy_mappings_from_user_terms": term_to_col
    }

    examples = [
        {"role": "user", "content": "Show average sales by region"},
        {"role": "assistant", "content": f'SELECT "region", AVG("sales") AS avg_sales FROM "{TABLE_NAME}" GROUP BY "region"'},
        {"role": "user", "content": "List top 3 customers by total sales"},
        {"role": "assistant", "content": f'SELECT "customer", SUM("sales") AS total_sales FROM "{TABLE_NAME}" GROUP BY "customer" ORDER BY total_sales DESC LIMIT 3'},
        {"role": "user", "content": "Counts by location with pivot on position"},
        {"role": "assistant", "content": f'SELECT "Location", "Position(Designation)" AS position, COUNT(*) AS cnt FROM "{TABLE_NAME}" GROUP BY "Location", "Position(Designation)"'},
    ]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Table: {TABLE_NAME}\nSchema & examples:\n{schema_note}"},
        *examples,
        {"role": "user", "content": f"Question: {body.question}\nRemember: only use schema columns. If pivot is needed, return tall form."}
    ]

    # --- Call Azure OpenAI ---
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            max_tokens=220
        )
        raw = resp.choices[0].message.content if resp and resp.choices else ""
    except Exception as e:
        return {"error": f"Azure OpenAI call failed: {e}"}

    sql = clean_sql(raw)
    if not sql:
        return {"error": "Empty SQL from model", "raw_output": raw}

    # --- Execute SQL ---
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn)
    except Exception as e:
        return {"error": str(e), "sql_attempted": sql, "raw_output": raw}

    # --- Build chart (if requested/keywords present) ---
    chart = None
    try:
        if wants_chart(body.question, body.want_chart):
            forced = detect_chart_type(body.question)
            chart = build_chart_from_df(df.copy(), forced_type=forced)

            # If "stacked" is in question, force stacked bars
            if chart and "stacked" in body.question.lower():
                if chart.get("type") == "bar":
                    chart.setdefault("options", {}).setdefault("scales", {})
                    chart["options"]["scales"]["x"] = {"stacked": True}
                    chart["options"]["scales"]["y"] = {"stacked": True}
    except Exception as e:
        chart = {"error": f"Chart build failed: {e}"}

    return {
        "sql": sql,
        "rows": int(len(df)),
        "result": df.to_dict(orient="records"),
        "mappings_used": term_to_col,
        "chart": chart
    }
