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
- **Sub-20ms Candidate Retrieval**: Achieved a **66% query latency reduction** (54.3ms $\rightarrow$ 18.6ms) on Cloud PostgreSQL using **Late Row Lookup (CTE)** and **B-Tree indexing**.
- **Jaccard Purity Index Ranking**: Solved multi-tag dilution by prioritizing emotional concentration over raw keyword counts.
- **2-Stage Hybrid Serving**: Combined ultra-fast indexed SQL candidate filtering (Stage 1) with real-time in-context LLM reasoning and 2nd-stage conversational refinement (Stage 2).

---

## System Architecture & Data Pipeline
```mermaid
flowchart TD
    User["👤 User Input\n(Narrative Story / Emotion Chips)"] --> FastAPI["⚡ FastAPI Server\n(Google Cloud Run Container)"]
    
    FastAPI -->|"1. Story Text"| GPT_Chips["🤖 OpenAI GPT-Luna\n(Auto-Suggests 3-5 Chip IDs)"]
    GPT_Chips --> DB
    
    FastAPI -->|"2. Selected Chip IDs"| DB["🗄️ Neon Cloud PostgreSQL (3NF Schema)\n• B-Tree Index: idx_song_chip_chip_id\n• Late Row Lookup (CTE)\n• Jaccard Purity Index Ranking\n⏱️ 18.6ms (66% Latency Reduction)"]
    
    DB -->|"Top 30 Candidates"| Curation["✨ Stage 2: LLM Narrative Curation\n• In-Context Reasoning (Top 3 Songs)\n• 1:1 English Curation Reasons\n• 2nd-Stage Conversational Refine"]
    
    Curation --> UI["🌐 Responsive Interactive Web UI\n(chip_eng.html)"]
```


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

## References
- Kapoor, S., Gil, A., Bhaduri, S., Mittal, A., & Mulkar, R. (2024). *Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling*. arXiv preprint. [https://arxiv.org/abs/2410.00290](https://arxiv.org/abs/2409.15626)
