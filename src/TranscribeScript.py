# Step 1. Run AWS Transcribe jobs for uploaded files in s3 bucket via boto3
import boto3
import time
import os

# Configuration
REGION = "us-east-1" 
BUCKET_NAME = "telehealth-capstone-data-alm"
RECORDINGS_PREFIX = "recordings/"
TRANSCRIPTS_PREFIX = "transcripts/"

s3 = boto3.client('s3', region_name=REGION)
transcribe = boto3.client('transcribe', region_name=REGION)

def start_transcription_jobs():
    # 1. List all files in the recordings folder
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=RECORDINGS_PREFIX)
    
    # Added a unique batch ID for this run to avoid "Job already exists"
    batch_id = "V2"

    for obj in response.get('Contents', []):
        file_key = obj['Key']
        if file_key == RECORDINGS_PREFIX: continue # Skip the folder itself
        
        file_name = file_key.split('/')[-1].replace('.', '-')
        job_name = f"Job-{file_name}-{batch_id}"
        media_uri = f"s3://{BUCKET_NAME}/{file_key}"
        
        print(f"--- Submitting: {job_name} ---")
        
        try:
            transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': media_uri},
                MediaFormat=file_key.split('.')[-1], 
                LanguageCode='en-US',
                OutputBucketName=BUCKET_NAME,
                OutputKey=f"{TRANSCRIPTS_PREFIX}{job_name}.json",
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 2 
                }
            )
        except Exception as e:
            print(f"Skipping {job_name}: {e}")

    # 2. Wait for completion
    print("\nWaiting for jobs to complete...")
    while True:
        status_response = transcribe.list_transcription_jobs(Status='IN_PROGRESS', MaxResults=100)
        if not status_response['TranscriptionJobSummaries']:
            print("All transcription jobs finished! Check your /transcripts folder in S3.")
            break
        print(f"Still processing {len(status_response['TranscriptionJobSummaries'])} jobs...")
        time.sleep(30)

if __name__ == "__main__":
    start_transcription_jobs()


# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Usage: After spending about three weeks on all of the other code, mainly by hand, I realized I needed script generation help for faster results. Used to create python script.
# Prompt: "Write me a python script with a goal of Converting audio to a structured format. The script needs to using Boto3:
# 1. Extract audio files from an s3 bucket "telehealth-capstone-data-alm/recordings/"
# 2. Start an Amazon Transcribe job with Speaker Diarization enabled and set to 2 speakers. Iterate through files in "telehealth-capstone-data-alm/recordings/"
# 3. Export transcription JSON results into an Excel table from exported Transcribe Json files saved in "telehealth-capstone-data-alm/transcripts/" to allow for manually verification if the AI heard the triggers correctly."
# -------------------------------------------------------------------------