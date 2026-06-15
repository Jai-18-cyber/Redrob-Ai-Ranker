import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def run_hackathon_ranker():
    print("1. Loading Pre-computed AI Vectors...")
    # Loading the data saved from the Phase 2 script
    with open("candidates_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    df = data["dataframe"]
    candidate_embeddings = data["embeddings"]

    print("2. Understanding the Job Description (JD)...")
    # This is the core summary of the Job Description extracted from the provided documents
    jd_text = "Senior AI Engineer. Deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, fine-tuning. Built recommendation systems, RAG, or search ranking. Pune/Noida or remote India."
    
    # Loading the same AI model to convert the JD into vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_vector = model.encode([jd_text])[0]

    print("3. Calculating AI Match Score...")
    # Cosine Similarity Formula (Mathematical logic to find how close the JD and Candidate vectors are)
    norm_jd = np.linalg.norm(jd_vector)
    norm_cands = np.linalg.norm(candidate_embeddings, axis=1)
    
    # Replacing 0 with a tiny number to prevent 'divide by zero' mathematical errors
    norm_cands = np.where(norm_cands == 0, 1e-10, norm_cands) 
    similarities = np.dot(candidate_embeddings, jd_vector) / (norm_cands * norm_jd)
    
    df['base_score'] = similarities

    print("4. Applying Secret Redrob Filters (Avoiding Traps!)...")
    # This function acts as our shield against the honeypot traps mentioned in the rules
    def calculate_final_score(row):
        score = row['base_score']
        
        # TRAP 1: Keyword Stuffers (e.g., Title is "HR" but skills list "AI")
        bad_titles = ['marketing', 'sales', 'hr', 'human resources', 'accountant', 'designer', 'customer support', 'content writer', 'operations']
        title_lower = str(row['title']).lower()
        if any(bad in title_lower for bad in bad_titles):
            score = score * 0.1  # Heavy penalty to drop them to the bottom of the list
            
        # TRAP 2: Inactive Candidates (Good profile but ghosting recruiters)
        resp_rate = row.get('response_rate', 0)
        # Active candidates get a bonus multiplier, inactive ones get a penalty
        score = score * (0.7 + (0.3 * resp_rate)) 
        
        return score

    df['final_score'] = df.apply(calculate_final_score, axis=1)

    print("5. Generating Final Top 100 List...")
    # Sorting the list from Highest score to Lowest score
    df_sorted = df.sort_values(by='final_score', ascending=False)
    
    # Hackathon rule: We strictly need exactly 100 rows.
    # (Since we are using the 50-candidate sample right now, it will just take what's available)
    top_candidates = df_sorted.head(100)

    # Creating the final CSV format strictly as per Hackathon rules
    final_rows = []
    rank = 1
    for idx, row in top_candidates.iterrows():
        cid = row['candidate_id']
        score = float(row['final_score'])
        exp = row['experience']
        title = row['title']
        resp = row['response_rate']
        
        # Rule: External LLM APIs are not allowed for ranking, so we use smart string formatting for the reasoning
        reasoning = f"Strong semantic match. {title} with {exp} yrs exp. Active candidate with {int(resp*100)}% response rate."
        
        final_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4), # Keeping it clean with 4 decimal places
            "reasoning": reasoning
        })
        rank += 1

    submission_df = pd.DataFrame(final_rows)
    
    # Saving the final submission file
    output_filename = "team_submission.csv"
    submission_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ BOOM! Hackathon task complete. '{output_filename}' is successfully generated!")
    print(submission_df.head(5))

if __name__ == "__main__":
    run_hackathon_ranker()