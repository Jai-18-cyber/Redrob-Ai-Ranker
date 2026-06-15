import json
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

def prepare_data_for_ai(file_path):
    print(f"Loading candidate data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        candidates = json.load(file)
        
    parsed_list = []
    for cand in candidates:
        candidate_id = cand.get("candidate_id")
        
        # Extract Profile details
        profile = cand.get("profile", {})
        title = profile.get("current_title", "Unknown")
        summary = profile.get("summary", "")
        exp = profile.get("years_of_experience", 0)
        
        # Extract Skills and convert them into a single comma-separated string
        skills_data = cand.get("skills", [])
        skills_text = ", ".join([s.get("name") for s in skills_data])
        
        # Extract Behavioral Signals (Crucial for filtering out inactive candidates)
        signals = cand.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0)
        last_active = signals.get("last_active_date", "")
        
        # THE SECRET MAGIC: Converting structured data into a single, rich contextual sentence for the AI to understand
        ai_text = f"Candidate is a {title} with {exp} years of experience. Skills include: {skills_text}. Profile Summary: {summary}"
        
        parsed_list.append({
            "candidate_id": candidate_id,
            "ai_text": ai_text,
            "title": title,
            "experience": exp,
            "response_rate": response_rate,
            "skills": skills_text
        })
        
    return pd.DataFrame(parsed_list)

if __name__ == "__main__":
    # 1. Load and format the raw JSON data
    print("1. Starting data preparation...")
    df = prepare_data_for_ai("sample_candidates.json")
    print(f"Successfully prepared {len(df)} candidates for AI processing.")

    # 2. Load the lightweight, open-source AI Model for embeddings
    print("\n2. Initializing the AI Model...")
    print("(If running for the first time, it might take a moment to load the model weights)")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Convert the generated text into dense vector representations (Embeddings)
    print("\n3. Converting candidate text into semantic AI vectors (Embeddings)...")
    text_list = df['ai_text'].tolist()
    embeddings = model.encode(text_list, show_progress_bar=True)

    # 4. Save the computed vectors to disk so the ranking script can run under 5 minutes
    print("\n4. Saving pre-computed vectors to 'candidates_data.pkl'...")
    data_to_save = {
        "dataframe": df,
        "embeddings": embeddings
    }
    
    with open("candidates_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)
        
    print("\n  'candidates_data.pkl' file has been successfully generated.")
