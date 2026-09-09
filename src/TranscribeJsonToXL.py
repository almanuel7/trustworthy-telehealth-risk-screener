# Step 2. Transfer downloaded JSON files to Readable Excel Files
import pandas as pd
import json
import os

# --- 1. CONFIGURATION ---
INPUT_DIR = "/Users/antoniomanuel/DTSC-691 Capstone Project/Transcribe JSON/"
OUTPUT_DIR = "/Users/antoniomanuel/DTSC-691 Capstone Project/JSON To Excel/"
MASTER_FILE = os.path.join(OUTPUT_DIR, "Master_Transcription_Dataset.xlsx")

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_transcribe_json(json_path, output_excel):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract Speaker Segments and Items
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
        
        rows.append({
            "Speaker": speaker,
            "Start Time": start_t,
            "Transcript": content.strip(),
            "Cousin's Validation": "" 
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    return df # Return the dataframe it can be used for the master list

# --- 2. PROCESSING LOOP ---
all_dataframes = []

for i in range(1, 46):
    input_filename = f"Job-Script{i}-wav-V2.json"
    output_filename = f"Session_{i}_Review.xlsx"
    
    # Create full absolute paths
    full_input_path = os.path.join(INPUT_DIR, input_filename)
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        print(f"Processing {input_filename}...")
        df_individual = parse_transcribe_json(full_input_path, full_output_path)
        
        # Add a column to identify which script this came from in the Master sheet
        df_individual['Source_Script'] = f"Script_{i}"
        all_dataframes.append(df_individual)
        
    except FileNotFoundError:
        print(f"File {input_filename} not found in {INPUT_DIR}. Skipping...")
    except Exception as e:
        print(f"An error occurred with {input_filename}: {e}")

# --- 3. MASTER CONCATENATION ---
if all_dataframes:
    print("\n--- Creating Master Excel Sheet ---")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Reorder columns to put Source_Script at the front
    cols = ['Source_Script'] + [c for c in master_df.columns if c != 'Source_Script']
    master_df = master_df[cols]
    
    master_df.to_excel(MASTER_FILE, index=False)
    print(f"Successfully saved Master Dataset to: {MASTER_FILE}")
else:
    print("No data was processed, Master file not created.")


# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Usage: After spending about three weeks on all of the other code, mainly by hand, I realized I needed script generation help for faster results. Used to create python script. The is the second script of two received.
# Prompt: "Write me a python script with a goal of Converting audio to a structured format. The script needs to using Boto3:
# 1. Extract audio files from an s3 bucket "telehealth-capstone-data-alm/recordings/"
# 2. Start an Amazon Transcribe job with Speaker Diarization enabled and set to 2 speakers. Iterate through files in "telehealth-capstone-data-alm/recordings/"
# 3. Export transcription JSON results into an Excel table from exported Transcribe Json files saved in "telehealth-capstone-data-alm/transcripts/" to allow for readable manually verification if the AI heard the triggers correctly."
# -------------------------------------------------------------------------