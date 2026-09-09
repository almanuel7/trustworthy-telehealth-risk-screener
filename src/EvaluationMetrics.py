# 7. Evaluation Metrics of Model
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from bert_score import score

# 1. Define Paths
directory_path = "/Users/antoniomanuel/DTSC-691 Capstone Project/Claude Analysis"
file_name = "Final_Claude_Analysis_Eval.xlsx"
file_path = os.path.join(directory_path, file_name)
cm_output_path = os.path.join(directory_path, "Confusion_Matrix.png")

def run_evaluation():
    print("Loading evaluation dataset...")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    # Ensure required columns exist
    required_cols = ['y_pred', 'y_true', 'ai_reasoning', 'reference_text']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'")
            return

    # Drop any rows where y_true or y_pred is missing
    df = df.dropna(subset=['y_true', 'y_pred'])
    
    # Standardize labels to string type
    y_true = df['y_true'].astype(str).tolist()
    y_pred = df['y_pred'].astype(str).tolist()

    # --- METRIC 1: Weighted F1-Score ---
    print("\n" + "="*40)
    print("1. CLASSIFICATION METRICS (F1-SCORE)")
    print("="*40)
    
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Overall Weighted F1-Score: {weighted_f1:.4f}\n")
    
    # Print the detailed breakdown per class
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred))


    # --- METRIC 2: Confusion Matrix ---
    print("Generating Confusion Matrix...")
    
    # Define clinical hierarchy for the matrix axes (Adjust these if your exact string labels differ)
    risk_labels = ['Low', 'Moderate', 'High', 'Imminent']
    
    # Fallback: if actual labels differ from the exact list above, use unique values found in data
    actual_labels = list(set(y_true + y_pred))
    labels_to_use = [l for l in risk_labels if l in actual_labels]
    if not labels_to_use: 
        labels_to_use = sorted(actual_labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels_to_use)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_to_use, yticklabels=labels_to_use)
    plt.ylabel('Actual Clinical Risk (y_true)')
    plt.xlabel('AI Predicted Risk (y_pred)')
    plt.title('Suicide Risk Triage: Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=300)
    print(f"Confusion Matrix saved to: {cm_output_path}")


    # --- METRIC 3: Semantic Similarity (BERTScore) ---
    print("\n" + "="*40)
    print("3. SEMANTIC SIMILARITY (BERTSCORE)")
    print("="*40)
    print("Initializing Roberta-Large model (This may take a moment to download on first run)...")
    
    # Handle missing text gracefully to avoid BERT crashes
    df['ai_reasoning'] = df['ai_reasoning'].fillna("No reasoning provided.")
    df['reference_text'] = df['reference_text'].fillna("No transcript available.")
    
    ai_texts = df['ai_reasoning'].tolist()
    ref_texts = df['reference_text'].tolist()

    # Calculate BERTScore
    # lang='en' defaults to roberta-large, an excellent model for clinical text mapping
    P, R, F1 = score(ai_texts, ref_texts, lang='en', verbose=True)

    print("\nBERTScore Results:")
    print(f"Mean Precision (Hallucination check): {P.mean().item():.4f}")
    print(f"Mean Recall (Information capture):    {R.mean().item():.4f}")
    print(f"Mean F1 (Overall semantic match):   {F1.mean().item():.4f}")

    print("\nEvaluation complete!")

if __name__ == "__main__":
    run_evaluation()

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Usage: Create a small script to perform evaluation metrics.
# Prompt: "Given 'Final_Claude_Analysis_Eval.xlsx' in path '/Users/antoniomanuel/DTSC-691 Capstone Project/Claude Analysis' 
# with the following columns 'Source_Script' ,'max_risk_score', 'ai_confidence', 'y_pred', 'y_true', 'ai_reasoning',and 'reference_text' create a 
# script for me that calculates Evaluation metrics: 1. Weighted F1-Score, 2. The Confusion Matrix, and 3. Semantic Similarity (BERTScore) (for hallucination check)."
# -------------------------------------------------------------------------