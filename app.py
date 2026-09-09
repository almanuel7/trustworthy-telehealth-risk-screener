from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import shutil
import boto3
import json
import time
import uuid
import os
import pandas as pd
import random
from transformers import pipeline

# --- 1. INITIALIZATION & CLIENTS ---
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
S3_BUCKET = os.environ.get("S3_BUCKET", "telehealth-capstone-data-alm")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3_client = boto3.client("s3", region_name=AWS_REGION)
transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

print("Loading RoBERTa Sentiment Model (This may take a moment)...")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
print("Model loaded successfully.")

# Initialize the FastAPI App
app = FastAPI()

# --- 2. TRIAGE FUNCTIONS ---
def check_therapist_intensity(df):
    crisis_keywords = [
        "safety plan", "emergency", "hospital", "stay with me", 
        "on the line", "ambulance", "911", "intervention", "harm yourself",
        "police", "don't hang up", "I need to stay with you", "keep you here with me", 
        "stay with you", "slow this down together", "intent" , "plan", "access", "preparation"
    ]
    therapist_text = " ".join(df[df['Speaker'] == 'spk_1']['Transcript'].astype(str)).lower()
    found_keywords = [word for word in crisis_keywords if word in therapist_text]
    return len(found_keywords) >= 2, found_keywords

def check_evasiveness(df):
    patient_turns = df[df['Speaker'] == 'spk_0']['Transcript'].astype(str).str.split().str.len()
    therapist_turns = df[df['Speaker'] == 'spk_1']['Transcript'].astype(str).str.split().str.len()
    
    if patient_turns.empty or therapist_turns.empty:
        return False, 0
    
    avg_p_len = patient_turns.mean()
    avg_t_len = therapist_turns.mean()
    is_evasive = (avg_t_len > avg_p_len * 4) and (avg_p_len < 4)
    return is_evasive, round(avg_p_len, 2)

def run_triage_checks(df):
    def get_score(text):
        if pd.isna(text) or str(text).strip() == "": return "NEUTRAL", 0.5
        res = sentiment_analyzer(str(text)[:1500])[0]
        return res['label'].upper(), res['score']

    df[['Sentiment', 'Score']] = df['Transcript'].apply(lambda x: pd.Series(get_score(x)))

    patient_talk = df[df['Speaker'] == 'spk_0']
    roberta_risk_score = 0.0
    
    if not patient_talk.empty:
        total_segments = len(patient_talk)
        neg_segments = patient_talk[patient_talk['Sentiment'] == 'NEGATIVE']
        density = len(neg_segments) / total_segments
        peak_intensity = neg_segments['Score'].max() if not neg_segments.empty else 0
        roberta_risk_score = (density * 0.6) + (peak_intensity * 0.4)

    is_crisis, triggers = check_therapist_intensity(df)
    is_evasive, p_avg_words = check_evasiveness(df)

    send_to_claude = (roberta_risk_score > 0.4) or is_crisis or is_evasive
    
    return {
        "RoBERTa_Risk": round(roberta_risk_score, 4),
        "Therapist_Crisis_Flag": is_crisis,
        "Evasive_Flag": is_evasive,
        "Send_To_Claude": send_to_claude
    }

# --- 3. AWS INFRASTRUCTURE FUNCTIONS ---
def upload_to_s3(file_path, bucket_name, object_name):
    try:
        with open(file_path, "rb") as f:
            s3_client.upload_fileobj(f, bucket_name, object_name)
        return f"s3://{bucket_name}/{object_name}"
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return None    

def start_transcription(job_name, file_uri):
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat='wav', 
            LanguageCode='en-US',
            OutputBucketName=S3_BUCKET,
            OutputKey=f"transcripts/{job_name}.json",
            Settings={'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 2}
        )
        return True
    except Exception as e:
        print(f"Transcription Start Error: {e}")
        return False

def get_transcription_result(job_name):
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']: break
        time.sleep(5)
    
    if job_status == 'COMPLETED':
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=f"transcripts/{job_name}.json")
        return json.loads(response['Body'].read().decode('utf-8'))
    return None

def parse_diarization(data):
    segments = data['results']['speaker_labels']['segments']
    items = data['results']['items']
    rows = []
    
    for seg in segments:
        speaker = seg['speaker_label']
        start_t = float(seg['start_time'])
        end_t = float(seg['end_time'])
        
        content = ""
        for item in items:
            if 'start_time' in item:
                item_start = float(item['start_time'])
                if start_t <= item_start <= end_t:
                    content += item['alternatives'][0]['content'] + " "
        
        rows.append({"Speaker": speaker, "Start Time": start_t, "Transcript": content.strip()})
    
    return pd.DataFrame(rows)

def invoke_bedrock(formatted_text):
    SYSTEM_PROMPT = """
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
    Return ONLY a JSON object with this exact structure:
    {
        "max_risk_score": 0.0-1.0,
        "ai_confidence": 0.0-1.0,
        "outcome": "Low / Moderate / High / Imminent",
        "step_by_step_logic": "Write structured clinical reasoning. Use numbered lists (1. Main Point) and bullet points (a. Subpoint) for high readability.",
        "visit_trend": [
            {
                "T": 1, 
                "cumulative_risk": 0.15,
                "trigger_words": ["ghost", "lonely"],
                "risk_label": "Low"
            }
        ],
        "detected_triggers": ["Label: Evidence Phrase"]
    }
    </output_format>
    </system_instructions>
    """
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Analyze the following diarized telehealth transcript:\n\n{formatted_text}"}],
        "temperature": 0.0
    }

    try:
        response = bedrock_client.invoke_model(modelId=MODEL_ID, body=json.dumps(payload))
        response_data = response.get('body').read().decode('utf-8')
        raw_text = json.loads(response_data)['content'][0]['text'].strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)
    
    except Exception as e:
        print(f"Bedrock Error: {e}")
        return {
            "max_risk_score": 0, "ai_confidence": 0.0, "outcome": "Processing Error",
            "step_by_step_logic": "Failed to parse Claude output.",
            "visit_trend": [], "detected_triggers": []
        }

# --- 4. FASTAPI ENDPOINTS ---

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/analyze_visit")
async def analyze_visit(
    patient_name: str = Form(...),
    visit_date: str = Form(...),
    audio_file: UploadFile = File(...)
):
    print(f"\n--- New Request Received: {patient_name} ---")
    
    file_id = f"{uuid.uuid4().hex[:8]}-{int(time.time())}"
    temp_file_path = f"temp_{file_id}.wav"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    try:
        print("Uploading to S3...")
        s3_uri = upload_to_s3(temp_file_path, S3_BUCKET, f"uploads/{file_id}.wav")
        if not s3_uri:
            raise Exception("Failed to upload file to S3")

        print("Starting Transcription...")
        job_name = f"Job-{file_id}"               
        start_transcription(job_name, s3_uri)
        transcribe_json = get_transcription_result(job_name)
        
        if not transcribe_json:
            raise Exception("Transcription failed or returned empty.")

        print("Running RoBERTa Triage...")
        df = parse_diarization(transcribe_json)
        triage_metrics = run_triage_checks(df)
        
        if not triage_metrics['Send_To_Claude']:
            result = {
                "max_risk_score": round(triage_metrics['RoBERTa_Risk'], 2), 
                "ai_confidence": 0.95,
                "outcome": "Low", 
                "step_by_step_logic": "1. RoBERTa Triage Clearance: Overall sentiment is stable. 2. No evasive patterns. 3. No professional crisis language detected.",
                "visit_trend": [
                    {"T": 5.0, "cumulative_risk": 0.1, "trigger_words": []},
                    {"T": 10.0, "cumulative_risk": 0.12, "trigger_words": []},
                    {"T": 15.0, "cumulative_risk": 0.08, "trigger_words": []},
                    {"T": 20.0, "cumulative_risk": 0.11, "trigger_words": []}
                ]
            }
        else:
            print("Flagged by RoBERTa. Sending to Claude for deep audit...")
            def format_row(row):
                t_min = int(row['Start Time'] // 60)
                t_sec = int(row['Start Time'] % 60)
                return f"[{t_min:02d}:{t_sec:02d}] {row['Speaker']}: {row['Transcript']}"
            
            formatted_text = "\n".join(df.apply(format_row, axis=1))
            result = invoke_bedrock(formatted_text)

        # --- DYNAMIC TIME SEGMENTER ---
        if "visit_trend" in result:
            # 1. Calculate the exact length of the session in minutes
            total_minutes = (df['Start Time'].max() + 10) / 60.0 if not df.empty else 60.0
            
            num_points = len(result["visit_trend"])
            if num_points > 0:
                # 2. Divide total time evenly by the number of points Claude analyzed
                interval = total_minutes / num_points
                
                for idx, point in enumerate(result["visit_trend"]):
                    # 3. Force T to be exact proportional timestamps (e.g. 0.33, 0.66)
                    point["T"] = round(interval * (idx + 1), 2)
                    if "minute" in point:
                        del point["minute"]
        # ------------------------------

        result["name"] = patient_name.upper()
        result["date"] = visit_date
        result["episode"] = f"O11{random.randint(10, 99)}"
        result["priority"] = 3 if result["outcome"].lower() == "low" else (1 if "imminent" in result["outcome"].lower() else 2)
        result["acknowledged"] = False

        print("Successfully packaged payload for frontend.")
        return result

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return {"error": str(e)}
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/clear_data")
async def clear_data():
    print("Clearing S3 Backend Data...")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="uploads/"):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="transcripts/"):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))