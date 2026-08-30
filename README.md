# 96-Chip Narrative Music Recommender (Lyrics-Driven)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon_Cloud-4169E1?logo=postgresql&logoColor=white)](https://neon.tech/)
[![Google Cloud Run](https://img.shields.io/badge/Google_Cloud_Run-Container-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5.6--Luna-412991?logo=openai&logoColor=white)](https://openai.com/)
[![GMU URSP](https://img.shields.io/badge/Research_Grant-GMU_URSP_Awardee-FFCC33?logo=georgemasonuniversity&logoColor=darkgreen)](https://oscar.gmu.edu/)

An end-to-end, full-stack AI music recommendation system that curates tracks from a **23,000+ song database** based on fine-grained **lyrical narrative arcs and psychological emotions**, rather than superficial audio/genre metadata. 

Awarded research grant funding by the **Undergraduate Research Scholars Program (URSP)** at George Mason University.

---

## Key Architectural Highlights

- **23,000+ Song Corpus**: Built an end-to-end ETL pipeline that collected tracks from curated Spotify playlists and scraped full lyrics from Genius, transforming raw unstructured text into structured narrative summaries and multi-theme mappings for 23,000+ songs.
- **96-Chip Emotion Taxonomy**: Hierarchical 2-tier ontology ($12\text{ Macro Domains} \times 8\text{ Sub-Nuance Micro Chips}$) constructed via LLM-enhanced semantic clustering, inspired by the topic modeling methodology proposed in QualIT [[Kapoor et al., 2024]]([https://arxiv.org/abs/2410.00290](https://arxiv.org/abs/2409.15626)).
- **66% Query Latency Reduction**: Achieved a query latency reduction (54.3ms → 18.6ms) on Cloud PostgreSQL using **Late Row Lookup (CTE)** and **B-Tree indexing**.
- **Jaccard Purity Index Ranking**: Solved multi-tag dilution by prioritizing emotional concentration over raw keyword counts.
- **2-Stage Hybrid Serving**: Combined ultra-fast indexed SQL candidate filtering (Stage 1) with real-time in-context LLM reasoning and 2nd-stage conversational refinement (Stage 2).

---

## End-to-End Data Pipeline (Batch ETL)

The offline data pipeline transforms raw, unstructured web data from multiple external sources into a highly structured 3NF relational database ready for real-time AI curation:

```mermaid
flowchart LR
    Step1["1️⃣ Spotify Scraping\n• Curated Playlists\n• Extracts Title & Artist"] --> Step2["2️⃣ Genius Scraping\n• Scrapes full raw lyrics\n• Matches Title + Artist"]
    
    Step2 --> Step3["3️⃣ Cleaning & Filtering\n• Strips live intros & ads\n• English language filter\n(28k to 23k Tracks)"]
    
    Step3 --> Step4["4️⃣ LLM Narrative Extraction\n(OpenAI GPT-5.6 Luna)\n• Generates story summaries\n• Extracts core theme tags"]
    
    Step4 --> Step5["5️⃣ 96-Chip Ontology Mapping\n• Semantic clustering\n• 12 Macro × 8 Micro Chips\n(QualIT Methodology)"]
    
    Step5 --> Step6[("6️⃣ Neon Cloud PostgreSQL\n• Ingests into 3NF schema\n• Songs, Lyrics & Mappings\n• Creates B-Tree Indexes")]
```

### Pipeline Workflow Breakdown:
1. **Track Metadata Extraction**: Extracted song titles and artist names across hundreds of curated Spotify playlists using Spotify API.
2. **Lyrics Scraping**: Automated Genius API/HTML scrapers to retrieve full raw lyrical texts matching track metadata.
3. **Data Scrubbing & Purity Filtering**: Stripped noisy bracket tags (`[Instrumental]`, `[Chorus]`) and filtered for pure English narrative songs (scrubbed ~5,000 noise records).
4. **LLM Narrative Summarization & Theme Extraction**: Utilized GPT-5.6 Luna to analyze raw lyrics, generating dense 2~3 sentence character-arc summaries along with multi-dimensional lyrical theme tags and keywords for every track.
5. **Hierarchical 96-Chip Mapping**: Applied semantic dictionary matching and clustering inspired by QualIT to categorize raw theme tokens into the structured 2-tier ontology ($12 \text{ Themes} \times 8 \text{ Nuance Chips}$).
6. **Relational DB Ingestion (3NF)**: Batch-loaded 23,000+ records into Neon PostgreSQL across 5 normalized tables (`categories`, `chips`, `songs`, `lyrics_info`, `song_chip_mappings`) and constructed B-Tree indexes for high-speed candidate retrieval.

---

## Engineering Challenges & Problem Solving

### 1. Robust Data Cleaning: Scrubbing 28,000+ Unstructured Raw Transcripts
* **The Problem**: Raw web-scraped lyrics from Genius and Spotify playlists contained severe data noise: non-musical TED talk transcripts, podcast dialogue scripts, pure instrumental tracks (`[Instrumental]`), live recording intros (`(Live at Wembley)`), and mixed non-English lyrics (Japanese, Spanish, etc,.). These noisy records corrupted emotional indexing and degraded recommendation quality.
* **The Solution**: Engineered a multi-stage automated batch preprocessing pipeline combining **heuristic regex cleaners, lyrical repetition thresholds, and language detection filters**. Successfully filtered out ~5,000 corrupted records, yielding a clean, high-purity corpus of **23,000+ narrative English songs**.

### 2. Semantic Collapse in Dense Vector Embeddings vs. Structured 96-Chip Ontology
* **The Problem (Baseline Failure: `Sentence-BERT all-MiniLM-L6-v2`)**: 
  * In our baseline experiments, standard dense embedding models (**`Sentence-BERT all-MiniLM-L6-v2`**) using Cosine Similarity failed significantly on lyrical text.
  * Poetic metaphors, repetitive choruses, and slang distorted the 384-dimensional vector space. Consequently, completely distinct emotional nuances—such as *"vengeful breakup anger"* vs. *"quiet melancholic resignation"*—were collapsed into the exact same dense vector cluster.
  * Complex narrative queries (e.g., *"quietly sacrificing for loved ones with unresolved guilt"*) consistently returned surface-level keyword false positives.
* **The Solution**: 
  * Replaced uninterpretable black-box vector distance with a **structured 96-Chip Hierarchical Emotion Taxonomy ($12 \text{ Domains} \times 8 \text{ Micro Chips}$)**, inspired by LLM topic modeling methodologies (QualIT).
  * Built a deterministic semantic mapping that translates thousands of raw lyrical tokens into discrete, noise-free ontology chips.

### 3. Multi-Tag Spam Dilution vs. Jaccard Purity Index Ranking
* **The Problem**: Ranking candidates solely by raw matching counts biased results toward songs with 15+ broad, generic tags, drastically lowering recommendation purity.
* **The Solution**: Formulated the **Jaccard Purity Index** to score songs based on emotional concentration:
    ```text
  Jaccard Purity Score = |Selected Chips ∩ Song Inherent Chips| / |Song Total Inherent Chips|
  ```
* **Result**: Prioritizes songs where the user's emotional query represents the **core lyrical focus** rather than a passing secondary mention.
### 4. PostgreSQL Query Latency Optimization (54.3ms $\rightarrow$ 18.6ms, 66% Speedup)
* **The Problem**: Profiling with `EXPLAIN ANALYZE` revealed that joining large `lyrics_summary` text columns across 23,000 rows during grouping and sorting created a memory bottleneck (137 buffer reads, 54.3ms latency).
* **The Solution**: Applied **Late Row Lookup (CTE)** to defer heavy text joins until after top-30 candidate IDs are filtered via B-Tree index:
  ```sql
  WITH matched_songs AS (
      SELECT m.song_id,
             COUNT(CASE WHEN m.chip_id = ANY(%s) THEN 1 END) AS match_count,
             ROUND(COUNT(CASE WHEN m.chip_id = ANY(%s) THEN 1 END)::numeric / COUNT(m.chip_id)::numeric, 3) AS jaccard_score
      FROM song_chip_mappings m
      WHERE m.song_id IN (SELECT song_id FROM song_chip_mappings WHERE chip_id = ANY(%s))
      GROUP BY m.song_id
      ORDER BY match_count DESC, jaccard_score DESC
      LIMIT 30
  )
  SELECT s.song_id, s.title, s.artist, l.lyrics_summary, l.raw_themes, w.jaccard_score
  FROM matched_songs w
  JOIN songs s ON w.song_id = s.song_id
  JOIN lyrics_info l ON w.song_id = l.song_id
  ORDER BY w.match_count DESC, w.jaccard_score DESC;


---
## Relational Database Schema (3NF)

The database is strictly normalized into **Third Normal Form (3NF)** on PostgreSQL to prevent data anomalies and maximize query indexing efficiency:
[categories] 1 ────< N [chips] 1 ────< N [song_chip_mappings] >──── N [songs] 1 ──── 1 [lyrics_info] 


* **`categories`**: Macro-level emotional themes (`category_id`, `name_en`, `name_ko`)
* **`chips`**: Micro-level emotion chips (`chip_id` [PK], `category_id` [FK], `name_en`, `name_ko`, `description`)
* **`songs`**: Core track entity metadata (`song_id` [PK], `title`, `artist`)
* **`lyrics_info`**: Detailed narrative and lyrical data (`song_id` [PK/FK], `lyrics_summary`, `raw_themes` [JSONB])
* **`song_chip_mappings`**: Many-to-Many junction table (`song_id`, `chip_id` [Compound PK])
  * **Indexed Column**: `CREATE INDEX idx_song_chip_chip_id ON song_chip_mappings(chip_id);`

---

## API Endpoints Specification

| Method | Endpoint | Description | Key Payload / Response |
|---|---|---|---|
| `GET` | `/health` | Cloud Run container health check | `{"status": "healthy"}` |
| `GET` | `/api/taxonomy` | Retrieves full 96-chip emotion hierarchy | `{"categories": [...]}` |
| `POST` | `/api/suggest-chips` | Auto-extracts 3~5 emotion chips from story text | `{ "text": "..." }` ➔ `{ "suggested_chip_ids": [...] }` |
| `POST` | `/api/recommend` | Stage 1 SQL retrieval + Stage 2 LLM curation | `{ "chip_ids": [...], "user_text": "...", "page": 1 }` |
| `POST` | `/api/refine` | 2nd-Stage conversational feedback & re-ranking | `{ "chip_ids": [...], "feedback": "More melancholic" }` |

---

## Getting Started & Local Development

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/caseyYtchoo/chip-music-recommendation-eng.git
cd chip-music-recommendation-eng
pip install -r requirements.txt psycopg2-binary
```

### 2. Configure Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DATABASE_URL="postgresql://user:password@host/dbname?sslmode=require"
```

### 3. Run Local Server
```bash
uvicorn chip_eng:app --host 0.0.0.0 --port 8080 --reload
```
Navigate to `http://localhost:8080` in your web browser.

---

## Docker Deployment (Google Cloud Run)

To build and deploy the containerized service directly to Google Cloud Run:

```bash
gcloud run deploy music-galaxy-en \
  --source . \
  --project gen-lang-client-0421276732 \
  --region asia-northeast3 \
  --set-env-vars="OPENAI_API_KEY=...,DATABASE_URL=..." \
  --allow-unauthenticated
```

---

## References

* Kapoor, S., Gil, A., Bhaduri, S., Mittal, A., & Mulkar, R. (2024). *Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling*. arXiv preprint. [https://arxiv.org/abs/2410.00290](https://arxiv.org/abs/2410.00290)

---

## Author & Academic Attribution

* **Author**: **Yeonseo (Casey) Tchoo**
* **Affiliation**: George Mason University Korea, Computational and Data Sciences (Honors College)
* **Funding**: Funded by the **GMU Undergraduate Research Scholars Program (URSP)**
* **Contact**: [caseytchoo@gmail.com](mailto:caseytchoo@gmail.com) | [LinkedIn Profile](https://www.linkedin.com/in/casey-tchoo-39474a392)
