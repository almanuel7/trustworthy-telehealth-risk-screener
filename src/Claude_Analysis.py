# 5. Claude Deep Analysis of 20 Random Samples
import boto3
import json
import pandas as pd
import os

# --- CONFIGURATION ---
REGION = "us-east-1"
TRIAGE_FILE = "/Users/antoniomanuel/DTSC-691 Capstone Project/JSON To Excel/Triage_Risk_List_Full_Context.xlsx"
OUTPUT_FOLDER = "/Users/antoniomanuel/DTSC-691 Capstone Project/Claude Analysis/"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize Bedrock Runtime
bedrock = boto3.client(service_name='bedrock-runtime', region_name=REGION)
# Define the Model ID for Claude Sonnet 4.6
MODEL_ID = "us.anthropic.claude-sonnet-4-6" 

# --- THE SYSTEM PROMPT (Crystalized Expert Logic) ---
SYSTEM_PROMPT = """
<system_instructions>
<role>Psychiatric Decision-Support Agent. Analyze transcripts for latent suicide risk via Interpersonal Theory (IPTS) and C-SSRS markers. You will receive a diarized transcript with timestamps.</role>

<context_key>
    - spk_0: PATIENT (Primary subject of risk assessment)
    - spk_1: PROFESSIONAL (Healthcare provider/Therapist)
</context_key>

<clinical_logic_framework>
    <construct_definitions>
        - THWARTED BELONGING: Social isolation, lack of reciprocal care, "ghost" metaphors, island/mainland detachment.
        - PERCEIVED BURDENSOMENESS: Belief that self is a liability/drain; "Better off without me," family better off financially/emotionally if I'm gone.
        - HOPELESSNESS (THE CATALYST): Belief that pain is permanent; "No light at the end of the tunnel," "Tried everything/nothing works."
        - ACQUIRED CAPABILITY: Habituation to pain/fear via past attempts, trauma exposure, or "practicing" motions; lowered fear of death/pain.
        - INTENT & PLAN (C-SSRS 4/5): Shift from "thinking" to "deciding/calm"; specific methods, timing, location, or "letters/goodbyes."
        - ACCESS & TOLERANCE: Possession of means (meds/weapons), scoped locations, high physical/psychological pain thresholds.
    </construct_definitions>

    <risk_tiering_logic>
        - TIER 1 (LOW): (Belonging OR Burdensomeness) AND (NO Capability). Focus: Interpersonal pain/loneliness.
        - TIER 2 (MODERATE): (Belonging AND Burdensomeness) AND (Hopelessness). Focus: Intersection of being alone + being a problem.
        - TIER 3 (HIGH): (Belonging AND Burdensomeness) AND (Hopelessness) AND (Capability OR Intent). Focus: Loss of biological "brakes" (fear).
        - TIER 4 (IMMINENT): ALL ABOVE AND (Access AND High Tolerance AND Specific Plan). Focus: Finality, resolution, and means-readiness.
    </risk_tiering_logic>
</clinical_logic_framework>

<output_format>
Return ONLY a JSON object with this exact structure of metrics for the Streamlit Heatmap:
{
    "max_risk_score": 0.0-10.0,
    "ai_confidence": 0.0-1.0,
    "outcome": "Low / Moderate / High / Imminent",
    "step_by_step_logic": "Clinical reasoning monologue",
    "visit_trend": [
        {
            "minute": 0.4, 
            "cumulative_risk": 0.15,
            "trigger_words": ["ghost", "lonely"],
            "risk_label": "Low"
        }
        // ... (one entry per 40 seconds of the session)
    ],
    "detected_triggers": ["Label: Evidence Phrase"]
}
</output_format>
</system_instructions>
"""

def analyze_with_claude(transcript_text):
    # Construct the payload for Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2500,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the following diarized telehealth transcript for risk markers:\n\n{transcript_text}"
            }
        ],
        "temperature": 0.0 # Essential for clinical consistency
    })

    try:
        response = bedrock.invoke_model(body=body, modelId=MODEL_ID)

        response_data = response.get('body').read().decode('utf-8')

        if not response_data:
            print("Received empty response from Bedrock.")
            return None
        
        response_body = json.loads(response_data)

        raw_text = response_body['content'][0]['text']
        # Clean the text (sometimes Claude adds markdown backticks ```json ... ```)
        clean_json = raw_text.strip()

        if clean_json.startswith("```json"):
            clean_json = clean_json.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_json)
    
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return None

# --- EXECUTION LOOP ---
def run_brain_part_2():
    df = pd.read_excel(TRIAGE_FILE)
    
    # We only process scripts where RoBERTa flagged 'Send_To_Claude' as True
    # Selected these 20 random scripts were 'Send_To_Claude' as True as a test to keep cost contained.
    high_risk_sessions = [
        'Script_4','Script_8','Script_10','Script_14','Script_15','Script_16','Script_17','Script_19','Script_20',
        'Script_23','Script_26','Script_27','Script_29','Script_34','Script_35','Script_37','Script_38','Script_40',
        'Script_43','Script_45'
    ]
    
    # This is the automated verion 
    # df[df['Send_To_Claude'] == True]['Source_Script'].unique()
    # I will only send 20 Random for Evaluation purposes
    
    all_claude_results = []

    for session_id in high_risk_sessions:
        print(f"Deep Analysis on: {session_id}")
        
        # FORMATTING THE TRANSCRIPT FOR CLAUDE
        # We concatenate [Timestamp] Speaker: Transcript
        session_data = df[df['Source_Script'] == session_id].copy()
        
        # Helper to format each row
        def format_row(row):
            # Formats as: [00:45] spk_0: I feel like a burden.
            time_min = int(row['Start Time'] // 60)
            time_sec = int(row['Start Time'] % 60)
            return f"[{time_min:02d}:{time_sec:02d}] {row['Speaker']}: {row['Transcript']}"

        formatted_transcript = "\n".join(session_data.apply(format_row, axis=1))

        result = analyze_with_claude(formatted_transcript)
        
        if result:
            result['Source_Script'] = session_id
            # We flatten the visit_trend into a string so it can be saved in Excel easily
            # The Streamlit app will json.loads() this later
            result['visit_trend_json'] = json.dumps(result['visit_trend'])
            del result['visit_trend'] # Remove the list to keep the Excel clean
            
            all_claude_results.append(result)

    # Save final results for the Streamlit Interface
    results_df = pd.DataFrame(all_claude_results)
    results_df.to_excel(os.path.join(OUTPUT_FOLDER, "Final_Claude_Analysis.xlsx"), index=False)
    print("--- Deep Clinical Analysis Saved ---")

if __name__ == "__main__":
    run_brain_part_2()


# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Usage: Create Optimal XML for Claude Call and ensuring the right elements return
# Prompt: "Write a python script to complete the deep reasoning part of the brain (Claude 4.6 Reasoning)
# which has a goal of deep clinical analysis using the expert guidance, found below:
# | Risk Level	| Boolean Logic Formula | C-SSRS Mapping | Label |
# |-------------|-----------------------|----------------|-------|
# | Green | (Belonging OR Burdensomeness) AND !Capability | 1 | Low |
# | Yellow | (Belonging AND Burdensomeness) AND Hopelessness | 2 | Moderate |
# | Orange | Yellow Logic AND (Capability OR Intent) | 3 - 4 | High |
# | Red | Orange Logic AND Plan AND Access AND High Tolerance | 5 + Behavior | Imminent | 
# Here is the contextual Logic provided by Professional:
# Low Risk (Monitor & Support)
# Logic: Vulnerability factors present, but no active desire or intent.
# (Thwarted Belonging OR Perceived Burdensomeness) AND (NO Capability)
# Assessment Focus: Identifying the start of interpersonal "pain."
# •	Interpersonal State: Mild Thwarted Belongingness (Loneliness).
# •	C-SSRS Level: 1 (Wish to be dead).
# •	Latent Triggers examples: 
# -	"I've been spending a lot of time alone lately."
# -	"I don't really feel like I fit in anywhere anymore."
# -	"Sometimes I just wish I didn't have to wake up tomorrow."
# -	"I’ve been staying in my room a lot more lately."
# -	"It feels like my friends have a group chat without me."
# -	"I'm just going through the motions every day."
# -	"I don't think I have a 'tribe' or a place where I fit."
# -	"I wonder if anyone would actually miss me if I moved away."
# -	"I’m tired of being the one who always reaches out first."
# -	"Lately, I just feel like a ghost in my own house."
# -	"My presence doesn't really seem to change the room."
# -	"I’ve lost that 'spark' or connection with my partner."
# -	"I wish I could just sleep for a year and skip all of this."
# •	Action: Provide resources, schedule follow-up, and focus on increasing social "Pulling Together" effects.
# Moderate Risk (In-Depth Assessment)
# Logic: Simultaneous presence of Belongingness and Burdensomeness issues.
# (Thwarted Belonging AND Perceived Burdensomeness) AND (Hopelessness)
# Assessment Focus: The intersection where the patient feels alone and like a problem that won't get better.
# •	Interpersonal State: Thwarted Belongingness + Perceived Burdensomeness.
# •	C-SSRS Level: 2 (Non-specific active suicidal thoughts).
# •	Latent Triggers:
# -	"My family would be so much better off if they didn't have to deal with me."
# -	"I'm just a drain on everyone's resources."
# -	"It feels like I'm a ghost; I'm there, but no one really sees or needs me."
# -	"My family is struggling because of my medical bills."
# -	"I’m a shadow of who I used to be; I’m just a burden now."
# -	"They’d be better off with the life insurance money than with me."
# -	"I’ve tried every therapy and med; nothing is ever going to work."
# -	"I’m just a drain on my parents' retirement."
# -	"No one should have to take care of me like this."
# -	"I’m a failure as a [provider/parent/spouse]."
# -	"The world is better off without my 'darkness' in it."
# -	"I can't see a single version of the future where I'm happy."
# -	"I’m just waiting for the inevitable; there’s no point in trying."
# •	Action: Safety planning, involve family/supports (if safe), and evaluate for Comorbid Mental Disorders like MDD or PTSD.
# High Risk (Urgent Intervention)
# Logic: Desire meets "Acquired Capability" (Lowered fear/High pain tolerance).
# (Active Desire) AND (Lowered Fear of Death OR Acquired Capability)
# Assessment Focus: The patient has the desire and has lost the biological "brakes" (fear) due to past trauma or attempts.
# •	Interpersonal State: Suicidal Desire + Acquired Capability (History of attempts or self-harm).
# •	C-SSRS Level: 3-4 (Active ideation with method/some intent).
# •	Latent Triggers:
# -	"I'm not afraid of dying anymore; I've been through worse."
# -	"I’ve already practiced how I would do it, just to see if I could."
# -	"Nothing hurts me as much as my own thoughts do."
# -	"Pain doesn't really bother me the way it used to.”
# -	"I’ve seen enough [violence/death/blood] to not be scared of it."
# -	"I've survived so much, the 'end' actually sounds peaceful."
# -	"I'm not a coward; I've got the guts to do what needs to be done."
# -	"I’ve been practicing how to keep my hand steady."
# -	"I've already stared down the barrel/ledge; the fear is gone."
# -	"After my last attempt, I realized it’s not as scary as people think."
# -	"I’m ready for whatever comes next."
# -	"I’ve habituated myself to the idea of not being here."
# -	"I feel a strange sense of 'calm' now that I've decided."
# •	Action: Immediate mental health consultation. Restrict Access to Lethal Means (firearms, medications). Establish Safety Plan.
# Imminent Risk (Emergency Protocol)
# Logic: Convergence of all components: Desire, Intent, Plan, and Capability.
# (Suicidal Intent) AND (Specific Plan) AND (Access to Means) AND (High Pain Tolerance)
# Assessment Focus: The "Perfect Storm" where all variables converge for action.
# •	Interpersonal State: Suicidal Intent + Specific Plan + Preparatory Behavior.
# •	C-SSRS Level: 5 (Specific plan and intent) + Preparatory Acts.
# •	Latent Triggers:
# -	"I've made sure all my affairs are in order, so no one has to clean up my mess."
# -	"I have the [Pills/Gun/Method] ready; I know exactly when I'm doing it."
# -	"I’ve said my goodbyes; I’m at peace with this decision."
# -	"I made sure my dog was rehomed with someone who loves him.”
# -	"I finally finished writing letters to everyone."
# -	“I went out and got exactly what I needed for the 'exit'."
# -	"I’ve picked the perfect spot where I won't be interrupted."
# -	"I’ve cleared my browser history and deleted my accounts."
# -	"Don't worry about my [debt/project/bills] anymore; it's handled."
# -	"I want to thank you for everything you tried to do for me." (Finality)
# -	"I have a 'back-up' plan if the first way doesn't work."
# -	"I’m doing this tonight while everyone is at the party."
# -	"I’ve finally found a way to stop the physical pain for good."
# •	Action: Do not leave the patient alone. Initiate emergency psychiatric evaluation/hospitalization. May need to contact local authorities for Wellness Check and Hospitalization Evaluation. Establish Safety Plan.
# In this process we need Bedrock Implementation for segments flagged after the RoBERTa Triage step, 
# send the text to Claude 4.6 Sonnet. Step up a system prompt using the XML tag format Claude prefers. 
# Include the triggers from our expert clinician using the following expert-defined triggers listed above
# analyze the provided transcript for passive suicide risk. Expilicitly mention thinking step-by-step."
# I would like to add structures so that claude is aware of the speaker roles 
# (maybe somehow concatenating 'Speaker' with 'Transcript' of the triage output file, along with also passing through 
# that 'spk_0' is patient and 'spk_1' is professional. Include timestamp along with functionality to create the desired
# heat map of visit. Please remember the metrics we would like to display within the web application stage including 
# the information needed to create the heat map visit trend of with trigger words included at their approximate time stamp.
# Please keep in mind we want to make the XML system prompt as concise as possible, but with enough instruction and context
# for Claude to be effective all while keeping the tokens to a minimum (for cost purposes). We still have to consider the length of the actual 
# scripts are no longer than ~2 minutes."
# -------------------------------------------------------------------------