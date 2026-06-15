import json
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

def prepare_data_for_ai(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        candidates = json.load(file)
        
    parsed_list = []
    for cand in candidates:
        candidate_id = cand.get("candidate_id")
        
        # Profile details
        profile = cand.get("profile", {})
        title = profile.get("current_title", "Unknown")
        summary = profile.get("summary", "")
        exp = profile.get("years_of_experience", 0)
        
        # Skills
        skills_data = cand.get("skills", [])
        skills_text = ", ".join([s.get("name") for s in skills_data])
        
        # Behavioral Signals
        signals = cand.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0)
        last_active = signals.get("last_active_date", "")
        
        # YAHI HAI HAMARA SECRET: Is ek line ko AI padhega aur samjhega
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
    # 1. Data load aur format karna
    df = prepare_data_for_ai("sample_candidates.json")
    print(f"Prepared {len(df)} candidates for AI processing.")

    # 2. AI Model Load karna
    print("\nDownloading/Loading AI Model (Pehli baar thoda time lag sakta hai internet speed ke hisaab se)...")
    # Ye ek bohot fast aur light-weight local AI model hai
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Text ko Vectors (Numbers) mein convert karna
    print("Converting candidate text into AI vectors (Embeddings)...")
    text_list = df['ai_text'].tolist()
    # Embeddings nikal rahe hain
    embeddings = model.encode(text_list, show_progress_bar=True)

    # 4. Save karna taaki baar-baar calculate na karna pade
    print("\nSaving vectors to 'candidates_data.pkl'...")
    data_to_save = {
        "dataframe": df,
        "embeddings": embeddings
    }
    
    with open("candidates_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)
        
    print("\n✅ Phase 2 Complete! 'candidates_data.pkl' file successfully ban gayi hai.")