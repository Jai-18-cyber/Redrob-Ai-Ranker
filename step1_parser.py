import json
import pandas as pd

def load_and_parse_candidates(file_path):
    print(f"Loading data from {file_path}...")
    
    # JSON file ko open karna
    with open(file_path, 'r', encoding='utf-8') as file:
        candidates = json.load(file)
        
    parsed_list = []
    
    # Har candidate ke data me se sirf kaam ki cheezein nikalna
    for cand in candidates:
        candidate_id = cand.get("candidate_id")
        
        # Profile details
        profile = cand.get("profile", {})
        title = profile.get("current_title", "Unknown")
        experience = profile.get("years_of_experience", 0)
        summary = profile.get("summary", "")
        
        # Skills (List of skills ko ek single text string mein convert karna)
        skills_data = cand.get("skills", [])
        skill_names = [skill.get("name") for skill in skills_data]
        skills_text = ", ".join(skill_names)
        
        # Behavioral Signals (Jo hackathon jeetne ke liye sabse zaroori hain)
        signals = cand.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0)
        notice_period = signals.get("notice_period_days", 90)
        
        # Ek dictionary mein save karna
        parsed_list.append({
            "candidate_id": candidate_id,
            "title": title,
            "experience_years": experience,
            "skills": skills_text,
            "summary": summary,
            "response_rate": response_rate,
            "notice_period": notice_period
        })
        
    # Is list ko pandas DataFrame (Table) mein convert kar dena
    df = pd.DataFrame(parsed_list)
    return df

# Code ko run karke test karte hain
if __name__ == "__main__":
    # Hum apni sample file use kar rahe hain test karne ke liye
    df_sample = load_and_parse_candidates("sample_candidates.json")
    
    print("\n✅ Data Successfully Parsed!")
    print(f"Total Candidates Found: {len(df_sample)}")
    print("-" * 50)
    print("Here is a sneak peek of the first 3 candidates:")
    # Displaying selected columns to check
    print(df_sample[['candidate_id', 'title', 'experience_years', 'response_rate']].head(3))