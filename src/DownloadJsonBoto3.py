# Step 3. Download transcribed files out of s3 bucket locally
import boto3
import os

# Configuration
BUCKET_NAME = "telehealth-capstone-data-alm"
S3_FOLDER = "transcripts/"
LOCAL_TARGET_DIR = "/Users/antoniomanuel/DTSC-691 Capstone Project/Transcribe JSON/"

# Initialize S3 Client
s3 = boto3.client('s3')

def download_transcripts():
    # 1. Ensure the local directory exists
    if not os.path.exists(LOCAL_TARGET_DIR):
        print(f"Creating directory: {LOCAL_TARGET_DIR}")
        os.makedirs(LOCAL_TARGET_DIR)

    # 2. List all objects in the transcripts/ folder
    print(f"Searching for JSON files in {BUCKET_NAME}/{S3_FOLDER}...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_FOLDER)

    files_downloaded = 0
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            s3_key = obj['Key']
            
            # Skip the folder marker itself and only download .json files
            if s3_key.endswith('.json'):
                # Extract just the filename from the S3 path
                filename = os.path.basename(s3_key)
                local_file_path = os.path.join(LOCAL_TARGET_DIR, filename)
                
                print(f"Downloading: {filename}")
                s3.download_file(BUCKET_NAME, s3_key, local_file_path)
                files_downloaded += 1

    print(f"\n--- Done! Total files downloaded: {files_downloaded} ---")

if __name__ == "__main__":
    download_transcripts()


# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Usage: Create a small script using boto3 to download transcribed files locallu.
# Prompt: "What script can be used to download all of the json files from transcripts/ within "telehealth-capstone-data-alm" 
# bucket name and download the files to "/Users/antoniomanuel/DTSC-691 Capstone Project/Transcribe JSON/"
# -------------------------------------------------------------------------