# Redrob AI Hackathon - Candidate Ranking Engine (Solo Submission)

## Architecture Overview
This project uses a Hybrid AI approach to rank candidates. Instead of simple keyword matching, it understands the semantic meaning of candidate profiles and Job Descriptions.

### How it works:
1. **Parser:** Extracts clean text and behavioral signals from the raw JSON.
2. **Pre-Compute Engine:** Converts candidate summaries into AI vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) and saves them locally. This allows the final ranker to run under 5 minutes on a CPU.
3. **Final Ranker & Trap Filter:** Matches vectors using Cosine Similarity, penalizes 'Honeypot' traps (e.g., HR titles with AI skills), and rewards active candidates based on their recruiter response rate.

## How to run locally
1. Install dependencies: `pip install pandas sentence-transformers numpy`
2. Run parser (optional): `python step1_parser.py`
3. Generate vectors: `python step2_precompute.py`
4. Get final ranking: `python step3_rank.py`
