# 4. RoBERTa Probabilistic Session Risk Score
import pandas as pd
from transformers import pipeline
import numpy as np
import os

# 1. Load RoBERTa
print("Loading RoBERTa model...")
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def check_therapist_intensity(group):
    """Checks if the professional (spk_1) is using crisis intervention language."""
    # Expert keywords provided by clinical guidelines
    crisis_keywords = [
        "safety plan", "emergency", "hospital", "stay with me", 
        "on the line", "ambulance", "911", "intervention", "harm yourself",
        "police", "don't hang up", "I need to stay with you", "keep you here with me", 
        "stay with you", "slow this down together", "intent" , "plan", "access", "preparation"
    ]
   
    therapist_text = " ".join(group[group['Speaker'] == 'spk_1']['Transcript'].astype(str)).lower()
    
    # Flag if 2 or more distinct crisis keywords are used
    found_keywords = [word for word in crisis_keywords if word in therapist_text]
    return len(found_keywords) >= 2, found_keywords

def check_evasiveness(group):
    """Checks if the patient (spk_0) is being clinically evasive (Short/Vague responses)."""
    patient_turns = group[group['Speaker'] == 'spk_0']['Transcript'].astype(str).str.split().str.len()
    therapist_turns = group[group['Speaker'] == 'spk_1']['Transcript'].astype(str).str.split().str.len()
    
    if patient_turns.empty or therapist_turns.empty:
        return False, 0
    
    avg_p_len = patient_turns.mean()
    avg_t_len = therapist_turns.mean()
    
    # Evasiveness Trigger: Therapist is talking 4x more than patient, 
    # and patient average response is under 4 words.
    is_evasive = (avg_t_len > avg_p_len * 4) and (avg_p_len < 4)
    return is_evasive, round(avg_p_len, 2)

def analyze_entire_corpus(input_path):
    df = pd.read_excel(input_path)
    
    # Run RoBERTa on every row first (Triage phase)
    print("Running line-by-line sentiment analysis...")
    def get_score(text):
        if pd.isna(text) or str(text).strip() == "": 
            return "NEUTRAL", 0.5
        res = sentiment_task(str(text)[:1500])[0] # truncate to 1500 chars since this is built for twitter
        return res['label'].upper(), res['score']

    df[['Sentiment', 'Score']] = df['Transcript'].apply(lambda x: pd.Series(get_score(x)))

    # 2. PROBABILISTIC SESSION WEIGHTING
    # We group by the Script ID to see the 'Whole'
    print("Calculating Interactional Risk Metrics...")
    session_summary = []
    
    for script_id, group in df.groupby('Source_Script'):

        # Filter for Patient only
        patient_talk = group[group['Speaker'] == 'spk_0']
        
        if not patient_talk.empty:
            total_segments = len(patient_talk)
            neg_segments = patient_talk[patient_talk['Sentiment'] == 'NEGATIVE']
            density = len(neg_segments) / total_segments
            peak_intensity = neg_segments['Score'].max() if not neg_segments.empty else 0
            roberta_risk_score = (density * 0.6) + (peak_intensity * 0.4)
        else:
            roberta_risk_score = 0.0

        # --- Check 2: Therapist Intensity ---
        is_crisis, triggers = check_therapist_intensity(group)
        
        # --- Check 3: Evasiveness ---
        is_evasive, p_avg_words = check_evasiveness(group)

        # --- FINAL TRIAGE DECISION ---
        # The 'Safety Net' Logic: Escalate if explicit distress OR therapist crisis mode OR clinical evasiveness
        send_to_claude = (roberta_risk_score > 0.4) or is_crisis or is_evasive
        
        session_summary.append({
            "Source_Script": script_id,
            "RoBERTa_Risk": round(roberta_risk_score, 4),
            "Therapist_Crisis_Flag": is_crisis,
            "Evasive_Flag": is_evasive,
            "Avg_Patient_Word_Count": p_avg_words,
            "Send_To_Claude": send_to_claude
        })

    summary_df = pd.DataFrame(session_summary)
    
    # 3. MERGE BACK
    # Add the decision back to the master list to know which rows to export
    final_df = df.merge(summary_df, on="Source_Script", how="left")
    return final_df, summary_df

# Run and Save
# Run the process
INPUT_PATH = "/Users/antoniomanuel/DTSC-691 Capstone Project/JSON To Excel/Master_Transcription_Dataset.xlsx"
OUTPUT_PATH = "/Users/antoniomanuel/DTSC-691 Capstone Project/JSON To Excel/Triage_Risk_List_Full_Context.xlsx"

master_df, summary = analyze_entire_corpus(INPUT_PATH)

master_df.to_excel(OUTPUT_PATH, index=False)

print(f"\n--- Triage Complete ---")
print(f"Total sessions flagged for Claude: {summary['Send_To_Claude'].sum()} out of {len(summary)}")
print(f"Flagged due to Evasiveness: {summary['Evasive_Flag'].sum()}")
print(f"Flagged due to Therapist Intervention: {summary['Therapist_Crisis_Flag'].sum()}")

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Usage: Complicated Implementation (within time constraints) of a number of concepts on the first pass (triage) to capture if patient 
# should be moved to the next stage (Claude Sonnet 4.6). This helps to keep cost contained while also ensuring that the output is readable.
# Prompt: "Write a python script to complete the triage first-pass of the brain (RoBERTa Classification)
# with a goal of utilizing RoBERTa as a high-speed  and inexpensive emotional screening. Include the following:
# a. Envoke the transformers library to load cardiffnlp/twitter-roberta-base-sentiment RoBERTa model.
# b. create script that calculates a Python Scripting: Run the transcripts through RoBERTa.
# c. line-by-line process the script segments with a sentiment outcome Postive, Neutral or Negative. Keep in mind, 
# I would like to have three criteria checks in the triage stage:
# 1. Probabilistic Session Risk Score sending line by line and then the score is calculated by grouping by first 'spk_0'
# (patient) then only calculating within the 'Source_Script' value.
# 2. Since Roberta is only focusing on speaker 0 it classifies it as a good sentiment therefore the negative probability 
# remains 0, we need to account for patients 'spk_1' who are using evasive language in comparison to the professional 'spk_1'. 
# Maybe a verbosity comparison of the number of words between professional and patient might be suffcient to account for this.
# 3. Lastly, a check for the professionals 'spk_1' responses employing a list of crisis/emergent keywords if the patient 'spk_1'
# is not directly using the normal self harm terms but the professional picks up on the patients reponses triggering a crisis
# mode. This should be an automatic send for deep analysis. key words should include 'stay with me', 'stay with you', 
# 'safety plan', etc.
# This process should occur on the JSON To Excel file Master_Transcription_Dataset.xlsx. Output file can be placed within the same
# folder within the same directory"
# -------------------------------------------------------------------------