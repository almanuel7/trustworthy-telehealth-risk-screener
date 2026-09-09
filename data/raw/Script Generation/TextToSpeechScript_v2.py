# Prework. After Generating scripts via ChatGPT utilize compute services from RunPod to create the actual .wav files of text scripts using randomized reference voices 
import os
import torch
import random
import soundfile as sf
import numpy as np
import gc
from qwen_tts import Qwen3TTSModel

# --- CONFIG ---
SCRIPTS_DIR = "/workspace/scripts"
REFS_DIR = "/workspace/references"
OUT_DIR = "/workspace/output_audio"
os.makedirs(OUT_DIR, exist_ok=True)

# Load Model
print("Loading Model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Define Reference Voices 
# Make sure these filenames match what you uploaded!
# I have removed personal information related to names and text ussed for voices
voices = [
    {"name": "Dr1", "audio": f"{REFS_DIR}/Doctor_One.wav", "text": "text from Doctor_One.wav file"},
    {"name": "Dr2", "audio": f"{REFS_DIR}/Doctor_Two.wav", "text": "text from Doctor_Two.wav file"},
    {"name": "Dr3", "audio": f"{REFS_DIR}/Doctor_Three.wav", "text": "text from Doctor_Three.wav file"},
    {"name": "Dr4", "audio": f"{REFS_DIR}/Doctor_Four.wav", "text": "text from Doctor_Four.wav file"},
    {"name": "Pt1", "audio": f"{REFS_DIR}/Patient_One.wav", "text": "text from Patient_One.wav file"},
    {"name": "Pt2", "audio": f"{REFS_DIR}/Patient_Two.wav", "text": "text from Patient_Two.wav file"},
    {"name": "Pt3", "audio": f"{REFS_DIR}/Patient_Three.wav", "text": "text from Patient_Three.wav file"},
    {"name": "Pt4", "audio": f"{REFS_DIR}/Patient_Four.wav", "text": "text from Patient_Four.wav file"},
]

# Processing Loop
script_files = [f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.txt')]
script_files.sort() # Process them in order

for filename in script_files:
    print(f"\n>>> Starting Script: {filename}")
    
    # Pick a consistent pair for this specific script
    dr = random.choice(voices[:4])
    pt = random.choice(voices[4:])
    
    all_segments = []
    
    with open(os.path.join(SCRIPTS_DIR, filename), 'r') as f:
        lines = [line.strip() for line in f if ":" in line]
        
    for i, line in enumerate(lines):
        role, text = line.split(":", 1)
        speaker = dr if "doctor" in role.lower() or "[professional]" in role.lower() else pt
        
        print(f"   Generating Line {i+1}/{len(lines)} ({role.strip()})")
        
        try:
            # Generate only this specific sentence
            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language="English",
                ref_audio=speaker["audio"],
                ref_text=speaker["text"]
            )
            all_segments.append(wavs[0])
            
            # Small silence (0.5s) between speakers for realism
            silence = np.zeros(int(sr * 0.75))
            all_segments.append(silence)
            
        except Exception as e:
            print(f"   ! Error on line {i+1}: {e}")
            continue

    # Combine all lines into one final file
    if all_segments:
        final_audio = np.concatenate(all_segments)
        output_path = os.path.join(OUT_DIR, f"{filename[:-4]}.wav")
        sf.write(output_path, final_audio, sr)
        print(f"Successfully saved: {output_path}")

    # --- VRAM RECOVERY ---
    # This clears the GPU memory after every script to keep it fast
    del all_segments
    torch.cuda.empty_cache()
    gc.collect()

print("\nAll tasks complete!")