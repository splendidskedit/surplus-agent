import asyncio
import json
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import ollama
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
DB_PATH = "liquidation.db"
OLLAMA_MODEL = "llama3.1"
SCAN_INTERVAL_SECONDS = 60

# Search dorks targeting industrial auction catalogs and LinkedIn buyers/sellers
DEFAULT_SEARCH_QUERIES = [
    {
        "category": "AUCTION_CATALOG",
        "platform": "BidSpotter / EquipNet",
        "query": '(site:bidspotter.com OR site:equipnet.com) ("surplus" OR "liquidation" OR "CNC" OR "transformer" OR "chiller" OR "generator")',
    },
    {
        "category": "INDIA_LIQUIDATION",
        "platform": "MSTC / IndiaMART / TradeIndia",
        "query": '(site:mstcindia.co.in OR site:indiamart.com) ("industrial surplus" OR "plant machinery sale" OR "scrap tender" OR "decommissioned")',
    },
    {
        "category": "LINKEDIN_SELLERS",
        "platform": "LinkedIn Sellers",
        "query": 'site:linkedin.com/posts ("surplus machinery" OR "plant liquidation" OR "idle equipment" OR "EXW" OR "asset disposal")',
    },
    {
        "category": "LINKEDIN_BUYERS",
        "platform": "LinkedIn Buyers",
        "query": 'site:linkedin.com/in ("Procurement Head" OR "Plant Manager" OR "Asset Recovery" OR "Used Machinery Buyer") ("Manufacturing" OR "Industrial")',
    },
]

app = FastAPI(title="macOS B2B Surplus Liquidation Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

agent_state = {
    "status": "Idle",
    "last_scan_utc": None,
    "next_scan_in_seconds": SCAN_INTERVAL_SECONDS,
    "total_scans_completed": 0,
    "active_query": "Waiting for cycle...",
    "custom_keyword": "",
}


# =====================================================================
# DATABASE LAYER (SQLite3 - Zero Cost, Serverless)
# =====================================================================
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog_leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            platform TEXT,
            lead_type TEXT,
            equipment_category TEXT,
            estimated_value_or_terms TEXT,
            location TEXT,
            contact_or_entity TEXT,
            urgency_score INTEGER,
            ai_summary TEXT,
            raw_snippet TEXT,
            discovered_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_lead(item: dict) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO catalog_leads (
                url, title, platform, lead_type, equipment_category,
                estimated_value_or_terms, location, contact_or_entity,
                urgency_score, ai_summary, raw_snippet, discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["url"],
                item["title"],
                item["platform"],
                item["lead_type"],
                item["equipment_category"],
                item["estimated_value_or_terms"],
                item["location"],
                item["contact_or_entity"],
                int(item["urgency_score"]),
                item["ai_summary"],
                item["raw_snippet"],
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )
        conn.commit()
        inserted = cursor.rowcount > 0
        return inserted
    except Exception:
        return False
    finally:
        conn.close()


# =====================================================================
# OLLAMA LOCAL AI EXTRACTION ENGINE
# =====================================================================
def analyze_with_ollama(title: str, snippet: str, url: str, default_platform: str) -> dict:
    """
    Uses local Ollama LLM to classify industrial liquidation leads into strict JSON.
    Includes a deterministic fallback if Ollama is busy or still downloading.
    """
    prompt = f"""
    You are an expert B2B Industrial Surplus Liquidation Broker operating on an EXW commission model.
    Analyze this web result and return ONLY a valid JSON object with these exact keys:
    - "lead_type": Must be one of ["SELLER_LISTING", "AUCTION_LOT", "BUYER_LEAD", "DIRECTORY"]
    - "equipment_category": Specific machinery type (e.g., "CNC Machining", "Power Transformers", "Packaging Line", "Heavy Industrial")
    - "estimated_value_or_terms": Extract price, bid status, or terms like "EXW / As-Is" (if unknown, write "EXW / Inquiry Required")
    - "location": City/Country mentioned, or "Global / Unspecified"
    - "contact_or_entity": Company name, auction house, or person's name/role from title
    - "urgency_score": Integer from 1 to 100 based on liquidation urgency (bankruptcy, closing auction, immediate disposal = 85-99)
    - "ai_summary": 1-sentence actionable brokerage summary (max 22 words).

    Input Data:
    Title: {title}
    Snippet: {snippet}
    URL: {url}
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
        )
        content = response["message"]["content"]
        parsed = json.loads(content)
        return {
            "lead_type": str(parsed.get("lead_type", "AUCTION_LOT"))[:30],
            "equipment_category": str(parsed.get("equipment_category", "Industrial Machinery"))[:50],
            "estimated_value_or_terms": str(parsed.get("estimated_value_or_terms", "EXW / As-Is"))[:60],
            "location": str(parsed.get("location", "Unspecified"))[:50],
            "contact_or_entity": str(parsed.get("contact_or_entity", default_platform))[:60],
            "urgency_score": max(1, min(100, int(parsed.get("urgency_score", 75)))),
            "ai_summary": str(parsed.get("ai_summary", snippet[:140]))[:180],
        }
    except Exception:
        # Deterministic zero-downtime fallback parser
        text_lower = f"{title} {snippet}".lower()
        if "linkedin.com/in" in url.lower() or "buyer" in text_lower or "procurement" in text_lower:
            lead_type = "BUYER_LEAD"
        elif "bidspotter" in url.lower() or "auction" in text_lower or "mstc" in url.lower():
            lead_type = "AUCTION_LOT"
        else:
            lead_type = "SELLER_LISTING"

        urgency = 85 if any(w in text_lower for w in ["liquidation", "urgent", "closing", "surplus", "auction"]) else 68
        return {
            "lead_type": lead_type,
            "equipment_category": "Industrial Surplus Asset",
            "estimated_value_or_terms": "EXW / As-Is Where-Is",
            "location": "Verify on Source",
            "contact_or_entity": title.split("-")[0].split("|")[0].strip()[:50],
            "urgency_score": urgency,
            "ai_summary": snippet[:150] if snippet else title,
        }


# =====================================================================
# LIVE INTERNET SCRAPER & MINUTE-BY-MINUTE BACKGROUND WORKER
# =====================================================================
def execute_discovery_cycle() -> None:
    agent_state["status"] = "Scanning Live Web & Running Ollama AI..."
    ddgs = DDGS()

    queries_to_run = list(DEFAULT_SEARCH_QUERIES)
    custom_kw = agent_state.get("custom_keyword", "").strip()
    if custom_kw:
        queries_to_run.insert(
            0,
            {
                "category": "CUSTOM_TARGET",
                "platform": f"Targeted ({custom_kw})",
                "query": f'("{custom_kw}") ("surplus" OR "liquidation" OR "used machinery" OR site:linkedin.com OR site:bidspotter.com OR site:indiamart.com)',
            },
        )

    for q_obj in queries_to_run:
        agent_state["active_query"] = q_obj["platform"]
        try:
            results = ddgs.text(
                q_obj["query"],
                region="wt-wt",
                safesearch="off",
                max_results=4,
            )
            if not results:
                continue

            for r in results:
                url = r.get("href", "").strip()
                title = r.get("title", "Untitled Industrial Listing").strip()
                body = r.get("body", "").strip()
                if not url:
                    continue

                ai_data = analyze_with_ollama(
                    title=title,
                    snippet=body,
                    url=url,
                    default_platform=q_obj["platform"],
                )

                record = {
                    "url": url,
                    "title": title,
                    "platform": q_obj["platform"],
                    "raw_snippet": body,
                    **ai_data,
                }
                insert_lead(record)
        except Exception as exc:
            print(f"[Search Warning] {q_obj['platform']}: {exc}")
            continue

    agent_state["total_scans_completed"] += 1
    agent_state["last_scan_utc"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    agent_state["active_query"] = "Idle (Waiting for next 60s tick)"
    agent_state["status"] = "Active · Monitoring 24/7"


def background_scheduler_loop() -> None:
    """Runs continuously in a daemon thread, triggering a catalog update every 60 seconds."""
    while True:
        execute_discovery_cycle()
        for remaining in range(SCAN_INTERVAL_SECONDS, 0, -1):
            agent_state["next_scan_in_seconds"] = remaining
            time.sleep(1)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    worker = threading.Thread(target=background_scheduler_loop, daemon=True)
    worker.start()


# =====================================================================
# API ENDPOINTS FOR MACOS FRONTEND
# =====================================================================
class KeywordUpdate(BaseModel):
    keyword: str


@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")


@app.get("/api/catalog")
def get_catalog(
    lead_type: Optional[str] = Query(default="ALL"),
    search: Optional[str] = Query(default=""),
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = "SELECT * FROM catalog_leads WHERE 1=1"
    params = []

    if lead_type and lead_type != "ALL":
        sql += " AND lead_type = ?"
        params.append(lead_type)

    if search:
        sql += " AND (title LIKE ? OR ai_summary LIKE ? OR equipment_category LIKE ? OR contact_or_entity LIKE ?)"
        wildcard = f"%{search}%"
        params.extend([wildcard, wildcard, wildcard, wildcard])

    sql += " ORDER BY id DESC LIMIT 120"
    cursor.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM catalog_leads")
    total_count = cursor.fetchone()[0]
    conn.close()

    return JSONResponse(
        {
            "agent_state": agent_state,
            "total_records": total_count,
            "items": rows,
        }
    )


@app.post("/api/target")
def set_custom_target(payload: KeywordUpdate):
    agent_state["custom_keyword"] = payload.keyword.strip()
    threading.Thread(target=execute_discovery_cycle, daemon=True).start()
    return {"status": "triggered", "custom_keyword": agent_state["custom_keyword"]}