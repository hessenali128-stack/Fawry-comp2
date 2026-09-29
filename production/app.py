from __future__ import annotations

import json
import os
import pickle
import sqlite3
import time
from pathlib import Path

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from recommender import Recommender
from ui.app import build_app
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = os.getenv("MODEL_BUNDLE", str(ROOT / "artifacts" / "model_bundle.joblib"))
DB_PATH = os.getenv("SQLITE_PATH", str(ROOT / "data" / "fawry.db"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

rec = Recommender(BUNDLE)

app = FastAPI(title="Fawry Recommendation API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, governorate TEXT, age_bucket TEXT, gender TEXT);
    CREATE TABLE IF NOT EXISTS offers (offer_id TEXT PRIMARY KEY, partner TEXT, category TEXT, description TEXT);
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        offer_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, created_at DESC);
    """)
    uf = rec.user_profiles.reset_index()
    for _, r in uf.iterrows():
        con.execute(
            "INSERT OR IGNORE INTO users(user_id,governorate,age_bucket,gender) VALUES(?,?,?,?)",
            (str(r.user_id), str(r.get("governorate", "")), str(r.get("age_bucket", "")), str(r.get("gender", "")))
        )
    for _, r in rec.offer_features.iterrows():
        con.execute(
            "INSERT OR IGNORE INTO offers(offer_id,partner,category,description) VALUES(?,?,?,?)",
            (str(r.offer_id), str(r.get("offer_partner_en", r.get("partner_san", ""))), str(r.get("cat_san", "")), str(r.get("gift_description", "")))
        )
    con.commit(); con.close()


class Click(BaseModel):
    user_id: str
    offer_id: str

try:
    rds = redis.Redis.from_url(REDIS_URL, decode_responses=False, socket_connect_timeout=1)
    rds.ping()
except Exception:
    rds = None
    print("Redis not reachable: using process memory fallback")

MEM = {}

def get_state(user_id):
    key = f"fawry:session:{user_id}"
    if rds:
        raw = rds.get(key)
        if raw:
            return pickle.loads(raw)
    return pickle.loads(MEM[key]) if key in MEM else rec.initial_session(user_id)


def set_state(user_id, state):
    key = f"fawry:session:{user_id}"
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    if rds:
        rds.setex(key, 86400, raw)
    else:
        MEM[key] = raw


def rec_cache_get(user_id):
    if not rds: return None
    raw = rds.get(f"fawry:recs:{user_id}")
    return json.loads(raw) if raw else None


def rec_cache_set(user_id, value):
    if rds:
        rds.setex(f"fawry:recs:{user_id}", 20, json.dumps(value))


def rec_cache_clear(user_id):
    if rds:
        rds.delete(f"fawry:recs:{user_id}")


@app.get("/api/health")
def health():
    return {"ok": True, "redis": bool(rds), "model": Path(BUNDLE).name, "features": rec.b["selected_features"]}


@app.get("/api/recommendations/{user_id}")
def recommendations(user_id: str, top_n: int = 10):
    cached = rec_cache_get(user_id)
    if cached is not None and top_n == 10:
        return {"user_id": user_id, "recommendations": cached, "cached": True}
    t0 = time.perf_counter()
    state = get_state(user_id)
    result = rec.recommend(user_id, state, top_n=max(1, min(top_n, 20)))
    rec_cache_set(user_id, result)
    return {"user_id": user_id, "recommendations": result, "latency_ms": round((time.perf_counter() - t0) * 1000, 3), "cached": False}


@app.post("/api/offers/click")
def click(payload: Click):
    if payload.offer_id not in rec.item_index:
        raise HTTPException(404, "Unknown offer_id")
    state = get_state(payload.user_id)
    state = rec.on_click(state, payload.offer_id)
    set_state(payload.user_id, state)
    rec_cache_clear(payload.user_id)
    con = db()
    con.execute("INSERT INTO events(user_id,offer_id,event_type,created_at) VALUES(?,?,?,?)",
                (payload.user_id, payload.offer_id, "click", time.time()))
    con.commit(); con.close()
    t0 = time.perf_counter()
    result = rec.recommend(payload.user_id, state, top_n=10)
    rec_cache_set(payload.user_id, result)
    return {"ok": True, "recommendations": result, "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}


@app.get("/api/features")
def features():
    return {"selected_features": rec.b["selected_features"], "importance": rec.b["selected_feature_importance"]}


init_db()
demo = build_app()
app = gr.mount_gradio_app(app, demo, path="/")
