import pandas as pd
import os
import re

# 1. Define the file paths and directories
excel_dir = "/Users/antoniomanuel/DTSC-691 Capstone Project/Claude Analysis"
excel_filename = "Final_Claude_Analysis_Eval.xlsx"
excel_path = os.path.join(excel_dir, excel_filename)

scripts_dir = "/Users/antoniomanuel/DTSC-691 Capstone Project/Script Generation/Scripts"

def extract_script_text(source_script_name):
    """
    Takes a Source_Script name (e.g., 'Script_4'), finds the corresponding .txt file 
    (e.g., 'Script4.txt'), reads the content, and strips out newlines.
    """
    # Handle empty/NaN cells gracefully
    if pd.isna(source_script_name):
        return ""
    
    # Format the string to match the text file naming convention (remove underscore, add .txt)
    txt_filename = str(source_script_name).replace("_", "") + ".txt"
    txt_filepath = os.path.join(scripts_dir, txt_filename)
    
    # Check if the text file actually exists before trying to open it
    if os.path.exists(txt_filepath):
        try:
            with open(txt_filepath, 'r', encoding='utf-8') as file:
                # Read the file and replace newlines with a space
                raw_text = file.read().replace('\n', ' ')
                
                # Clean up any accidental double-spaces created by replacing newlines
                cleaned_text = re.sub(' +', ' ', raw_text).strip()
                
                return cleaned_text
        except Exception as e:
            print(f"Error reading {txt_filename}: {e}")
            return ""
    else:
        print(f"Warning: File {txt_filename} not found in directory.")
        return ""

def process_evaluation_file():
    print(f"Loading Excel file: {excel_filename}...")
    
    try:
        # 2. Load the evaluation dataset
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {excel_path}")
        return

    # Verify that the 'Source_Script' column exists
    if 'Source_Script' not in df.columns:
        print("Error: 'Source_Script' column not found in the Excel file.")
        return

    print("Mapping files and extracting text...")
    
    # 3. Apply the extraction function to create the new 'reference_text' column
    df['reference_text'] = df['Source_Script'].apply(extract_script_text)
    
    # 4. Save the updated dataframe back to the original file
    print(f"Saving updates back to {excel_filename}...")
    df.to_excel(excel_path, index=False)
    
    print("Process completed successfully!")

# Execute the script
if __name__ == "__main__":
    process_evaluation_file()