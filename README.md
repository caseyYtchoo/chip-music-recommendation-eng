# 🌌 96-Chip Narrative Music Recommender (Lyrics-Driven)

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
- **96-Chip Emotion Taxonomy**: Hierarchical 2-tier ontology ($12\text{ Macro Domains} \times 8\text{ Sub-Nuance Micro Chips}$) constructed via LLM-enhanced semantic clustering, inspired by the topic modeling methodology proposed in QualIT [[Kapoor et al., 2024]](https://arxiv.org/abs/2410.00290).
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





## References
- Kapoor, S., Gil, A., Bhaduri, S., Mittal, A., & Mulkar, R. (2024). *Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling*. arXiv preprint. [https://arxiv.org/abs/2410.00290](https://arxiv.org/abs/2410.00290)
