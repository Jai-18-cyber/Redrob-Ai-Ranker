import json
import pandas as pd

def load_and_parse_candidates(file_path):
    print(f"Loading candidate data from {file_path}...")
    
    # Open and read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        candidates = json.load(file)
        
    parsed_list = []
    
    # Extract only the necessary and highly relevant fields from each candidate's data
    for cand in candidates:
        candidate_id = cand.get("candidate_id")
        
        # 1. Extract Basic Profile Details
        profile = cand.get("profile", {})
        title = profile.get("current_title", "Unknown")
        experience = profile.get("years_of_experience", 0)
        summary = profile.get("summary", "")
        
        # 2. Extract Skills (Converting the list of dictionaries into a single comma-separated string)
        skills_data = cand.get("skills", [])
        skill_names = [skill.get("name") for skill in skills_data]
        skills_text = ", ".join(skill_names)
        
        # 3. Extract Behavioral Signals (Crucial for filtering out inactive/fake candidates)
        signals = cand.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0)
        notice_period = signals.get("notice_period_days", 90)
        
        # Append the cleaned data as a dictionary to our list
        parsed_list.append({
            "candidate_id": candidate_id,
            "title": title,
            "experience_years": experience,
            "skills": skills_text,
            "summary": summary,
            "response_rate": response_rate,
            "notice_period": notice_period
        })
        
    # Convert the extracted list of dictionaries into a Pandas DataFrame (Table format)
    df = pd.DataFrame(parsed_list)
    return df

if __name__ == "__main__":
    # Test the parser using the sample dataset
    print("Initializing Data Parser...")
    df_sample = load_and_parse_candidates("sample_candidates.json")
    
    print("\n Data Successfully Parsed!")
    print(f"Total Candidates Found: {len(df_sample)}")
    print("-" * 50)
    print("Here is a sneak peek of the first 3 candidates:")
    
    # Displaying selected columns to verify the output
    print(df_sample[['candidate_id', 'title', 'experience_years', 'response_rate']].head(3))
