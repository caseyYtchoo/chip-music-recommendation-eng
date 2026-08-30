import os
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI

app = FastAPI(title="96-Chip Narrative Music Recommender API (EN)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 🔑 1. Security & Configuration (Zero Key Exposure)
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
GPT_MODEL = "gpt-5.6-luna"

DATABASE_URL = os.getenv("DATABASE_URL", "")

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=5)

# =========================================================
# 📂 2. Dynamic Taxonomy & Chip Dictionary Loader
# =========================================================
taxonomy_cache = {"categories": []}
chip_dict = {}

def get_taxonomy_data():
    global taxonomy_cache, chip_dict
    if taxonomy_cache.get("categories"):
        return taxonomy_cache, chip_dict
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM categories ORDER BY category_id;")
        cats = cur.fetchall()
        cur.execute("SELECT * FROM chips ORDER BY category_id, chip_id;")
        chips = cur.fetchall()
        cur.close()
        conn.close()

        categories_list = []
        for c in chips:
            tag_id = str(c["chip_id"]).lower()
            chip_dict[tag_id] = {
                "tag_id": tag_id,
                "chip_id": tag_id,
                "name_ko": c["name_ko"],
                "name_en": c["name_en"],
                "description": c.get("description", "") or ""
            }

        for cat in cats:
            cat_chips = []
            for c in chips:
                if c["category_id"] == cat["category_id"]:
                    tid = str(c["chip_id"]).lower()
                    cat_chips.append({
                        "tag_id": tid,
                        "chip_id": tid,
                        "id": tid,
                        "name_ko": c["name_ko"],
                        "name_en": c["name_en"],
                        "description": c.get("description", "") or ""
                    })
            categories_list.append({
                "category_id": cat["category_id"],
                "category_name_ko": cat["name_ko"],
                "category_name_en": cat["name_en"],
                "sub_themes": cat_chips
            })

        taxonomy_cache = {"categories": categories_list}
    except Exception as e:
        print(f"⚠️ DB Taxonomy Query Notice: {e}")
    return taxonomy_cache, chip_dict

# =========================================================
# 🤖 3. Narrative-Based Emotion Chip Suggestion (LLM)
# =========================================================
def suggest_chips_from_text(user_text: str) -> List[str]:
    _, cd = get_taxonomy_data()
    chips_summary = "\n".join([f"{c['tag_id']}: [{c['name_en']}] - {c['description']}" for c in cd.values()])
    prompt = f"""You are a senior music psychologist and narrative curator. Read the user's emotional story and select the 3 to 5 most fitting emotion chips.
Return ONLY the matching tag_ids separated by commas.

[Available Emotion Chips]
{chips_summary}

[User Narrative / Story]
{user_text}

[Output Format Example] tag_01, tag_08, tag_45"""

    try:
        if not client:
            return ["tag_01", "tag_05", "tag_20"]
        response = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}])
        found_tags = re.findall(r'\b(tag_\d+)\b', response.choices[0].message.content.strip().lower())
        return list(set(t for t in found_tags if t in cd))[:5]
    except Exception:
        return ["tag_01", "tag_05", "tag_20"]

# =========================================================
# 🎯 4. Candidate Retrieval via Late Row Lookup & Jaccard Ranking
# =========================================================
def find_songs_by_chips_from_db(selected_chip_ids: List[str], top_k: int = 30):
    _, cd = get_taxonomy_data()
    selected_chip_ids = [c.lower() for c in selected_chip_ids if c.lower() in cd]
    if not selected_chip_ids:
        return []

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
        WITH matched_songs AS (
            SELECT 
                m.song_id,
                COUNT(CASE WHEN m.chip_id = ANY(%s) THEN 1 END) AS match_count,
                ARRAY_AGG(m.chip_id) FILTER (WHERE m.chip_id = ANY(%s)) AS matched_chip_ids,
                ROUND(
                    COUNT(CASE WHEN m.chip_id = ANY(%s) THEN 1 END)::numeric / 
                    COUNT(m.chip_id)::numeric, 
                    3
                ) AS jaccard_score
            FROM song_chip_mappings m
            WHERE m.song_id IN (
                SELECT song_id FROM song_chip_mappings WHERE chip_id = ANY(%s)
            )
            GROUP BY m.song_id
            ORDER BY match_count DESC, jaccard_score DESC
            LIMIT %s
        )
        SELECT 
            s.song_id, s.title, s.artist, l.lyrics_summary, l.raw_themes,
            w.match_count AS score, w.jaccard_score, w.matched_chip_ids
        FROM matched_songs w
        JOIN songs s ON w.song_id = s.song_id
        JOIN lyrics_info l ON w.song_id = l.song_id
        ORDER BY w.match_count DESC, w.jaccard_score DESC;
        """
        cur.execute(query, (selected_chip_ids, selected_chip_ids, selected_chip_ids, selected_chip_ids, top_k))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        formatted_songs = []
        for r in rows:
            matched_chip_names = []
            for cid in (r["matched_chip_ids"] or []):
                if cid in cd:
                    matched_chip_names.append(cd[cid]["name_en"])
            raw_themes = r["raw_themes"] if isinstance(r["raw_themes"], list) else []
            formatted_songs.append({
                "song_id": r["song_id"],
                "artist": r["artist"],
                "title": r["title"],
                "raw_themes": raw_themes,
                "lyrics_summary": r["lyrics_summary"],
                "score": r["score"],
                "jaccard_score": float(r["jaccard_score"] or 0),
                "matched_chips": list(set(matched_chip_names))
            })
        return formatted_songs
    except Exception as e:
        print(f"❌ DB Query Error: {e}")
        return []

# =========================================================
# 📝 5. AI Narrative Curation & Reasoning (Structured JSON Output)
# =========================================================
def generate_reasons(selected_chips_en, target_3_songs, feedback_text=""):
    def get_single_reason(song):
        chips_str = ", ".join(selected_chips_en)
        raw_themes = song.get("raw_themes", [])
        themes_str = ", ".join(raw_themes) if isinstance(raw_themes, list) else str(raw_themes)
        feedback_prompt = f"\nUser Additional Nuance / Context: \"{feedback_text}\"" if feedback_text else ""
        
        prompt = f"""You are a professional music curator and playlist editor. Analyze the track details below and return your recommendation strictly in pure JSON format.

[Track Details]
- Selected Emotion Chips: [{chips_str}]{feedback_prompt}
- Song: {song['artist']} - \"{song['title']}\"
- Inherent Themes: [{themes_str}]
- Lyrical Story Summary: {song['lyrics_summary']}

[Output Requirements: Return ONLY the JSON schema below]
{{
  "reason": "1-2 sentences of thoughtful, insightful curation reason in fluent English explaining why this song perfectly resonates with the selected emotions and user context.",
  "lyrics_summary_en": "2-3 sentences summarizing the song's lyrical journey, psychological tension, and message in evocative English.",
  "english_themes": ["Theme 1", "Theme 2", "Theme 3"]
}}"""

        try:
            if not client:
                return {
                    **song, 
                    "reason": "This song deeply resonates with your selected emotional nuance.", 
                    "lyrics_summary_en": song.get("lyrics_summary", "An expressive lyrical narrative exploring emotional depth."),
                    "lyrics_summary_ko": song.get("lyrics_summary", "An expressive lyrical narrative."),
                    "bilingual_themes": [str(t).title() for t in raw_themes]
                }
            
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content.strip())
            
            reason = str(data.get("reason", "A track that deeply captures the requested emotional narrative.")).strip()
            summary_en = str(data.get("lyrics_summary_en", song.get("lyrics_summary", ""))).strip()
            english_themes = data.get("english_themes", [])
            if not isinstance(english_themes, list):
                english_themes = [str(english_themes)]
            
            # Clean markdown artifacts
            reason = re.sub(r'[\*\#]', '', reason).strip()
            summary_en = re.sub(r'[\*\#]', '', summary_en).strip()

            clean_themes = []
            theme_pool = english_themes if english_themes else raw_themes
            for raw_t in theme_pool:
                clean_t = re.sub(r'[\*\#]', '', str(raw_t)).strip().title()
                if clean_t:
                    clean_themes.append(clean_t)
            
            return {
                **song, 
                "reason": reason, 
                "lyrics_summary_en": summary_en,
                "lyrics_summary_ko": summary_en,  # Backward compatibility for UI
                "bilingual_themes": clean_themes
            }
        except Exception as e:
            return {
                **song, 
                "reason": "A standout track that powerfully conveys the speaker's emotional state and lyrical story.", 
                "lyrics_summary_en": song.get("lyrics_summary", "A lyrical narrative centered on emotional expression."),
                "lyrics_summary_ko": song.get("lyrics_summary", "A lyrical narrative."),
                "bilingual_themes": [str(t).replace("#", "").title() for t in raw_themes]
            }

    with ThreadPoolExecutor(max_workers=3) as executor:
        return list(executor.map(get_single_reason, target_3_songs))

# =========================================================
# 💬 6. Conversational 2nd-Stage Feedback Fine-Tuning
# =========================================================
def refine_by_feedback(candidates, feedback_text, selected_chips_en):
    pool = candidates[:25]
    song_summaries = "\n".join([f"{idx+1}. [{s['artist']} - {s['title']}] {s['lyrics_summary']}" for idx, s in enumerate(pool)])
    prompt = f"User Feedback: \"{feedback_text}\"\nSelect the 3 best song indices from the candidate summaries below, separated by commas:\n{song_summaries}"
    try:
        if not client:
            return generate_reasons(selected_chips_en, pool[:3], feedback_text)
        response = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}])
        nums = [int(n) for n in re.findall(r'\b\d+\b', response.choices[0].message.content) if 1 <= int(n) <= len(pool)]
        selected_3 = [pool[i-1] for i in nums[:3]] if len(nums) >= 3 else pool[:3]
    except Exception:
        selected_3 = pool[:3]
    return generate_reasons(selected_chips_en, selected_3, feedback_text)

# =========================================================
# 🌐 7. API Routes & Endpoints
# =========================================================
class SuggestRequest(BaseModel):
    text: str

class RecommendRequest(BaseModel):
    chip_ids: List[str]
    user_text: Optional[str] = ""
    page: int = 1
    shuffle: bool = False

class RefineRequest(BaseModel):
    chip_ids: List[str]
    user_text: Optional[str] = ""
    feedback: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/taxonomy")
def api_get_taxonomy():
    tax, _ = get_taxonomy_data()
    return tax

@app.post("/api/suggest-chips")
def api_suggest_chips(req: SuggestRequest):
    return {"suggested_chip_ids": suggest_chips_from_text(req.text)}

@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    cands = find_songs_by_chips_from_db(req.chip_ids, top_k=30)
    if not cands:
        return {"songs": [], "page": req.page, "total_candidates": 0}
    if req.page == 1 and not req.shuffle:
        target_songs = cands[:3]
    elif req.shuffle:
        pool = cands[:min(len(cands), 20)]
        target_songs = random.sample(pool, min(len(pool), 3))
    else:
        start_idx = (req.page - 1) * 3
        target_songs = cands[start_idx:start_idx + 3] if start_idx < len(cands) else cands[:3]

    _, cd = get_taxonomy_data()
    selected_en = [cd[cid]["name_en"] for cid in req.chip_ids if cid in cd]
    return {"songs": generate_reasons(selected_en, target_songs, req.user_text), "page": req.page, "total_candidates": len(cands)}

@app.post("/api/refine")
def api_refine(req: RefineRequest):
    cands = find_songs_by_chips_from_db(req.chip_ids, top_k=30)
    if not cands:
        return {"songs": []}
    _, cd = get_taxonomy_data()
    selected_en = [cd[cid]["name_en"] for cid in req.chip_ids if cid in cd]
    return {"songs": refine_by_feedback(cands, req.feedback, selected_en), "feedback": req.feedback}

@app.get("/")
def serve_index():
    for f in ["chip_eng.html", "chip.html", "index.html"]:
        if os.path.exists(f):
            return FileResponse(f)
    return {"message": "Music Galaxy English Server Online"}