# Trustworthy Telehealth Risk Screener

A capstone project (DTSC 691) demonstrating an AI-assisted clinical decision-support pipeline that screens telehealth therapy session recordings for suicide-risk indicators, using a two-stage triage: a lightweight sentiment model for fast first-pass screening, and Claude (via Amazon Bedrock) for deep clinical review of anything flagged.

**Disclaimer:** This is an academic demo built on synthetic, scripted session audio (see `data/`) generated from public research datasets. It is not a validated clinical tool and must not be used to make real patient-care decisions.

## How it works

1. A therapist/clinician uploads a session audio file (`.wav`) and basic visit info through the web UI (`index.html`).
2. The FastAPI backend (`app.py`) uploads the audio to Amazon S3 and starts an Amazon Transcribe job with speaker diarization.
3. The diarized transcript is run through a RoBERTa sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) plus rule-based checks (crisis language from the clinician, evasive answer patterns from the patient) to compute a fast triage risk score.
4. Sessions that clear this first pass are marked low-risk immediately. Sessions that don't are sent to Claude (via Amazon Bedrock) with a structured clinical prompt based on the Interpersonal Theory of Suicide (IPTS) and C-SSRS risk markers, which returns a tiered risk assessment (Low / Moderate / High / Imminent) with supporting reasoning.
5. The dashboard renders the result: risk tier, trend over the course of the visit, and the specific language that drove the assessment.

## Repository layout

- `index.html` — the frontend dashboard (static; served by the backend)
- `app.py` — FastAPI backend: `/api/analyze_visit` and `/api/clear_data` endpoints, AWS integration, triage logic
- `requirements.txt` — Python dependencies for `app.py`
- `src/` — offline research and data-processing scripts used while building the pipeline (synthetic script generation, transcription batch jobs, model evaluation). These reference the original local development machine's file paths and are not required to run the live app — they're included for transparency into the project's development process.
- `data/` — sample synthetic session data (scripted audio, transcripts, evaluation spreadsheets) used during development and evaluation. Not required at runtime; the live app processes whatever audio a user uploads.

## Running locally

Requires Python 3.10+ and AWS credentials with access to S3, Transcribe, and Bedrock (see below).

```bash
pip install -r requirements.txt

export S3_BUCKET=your-bucket-name
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
# AWS credentials via `aws configure`, environment variables, or an IAM role

python app.py
```

Then open `http://127.0.0.1:8000` in a browser (not the `index.html` file directly — the frontend calls the backend over relative `/api/...` paths, so it must be loaded through the running server).

## AWS setup required

- An S3 bucket for uploaded audio and Transcribe output.
- Amazon Transcribe enabled in your region.
- Amazon Bedrock model access granted for the Claude model referenced by `BEDROCK_MODEL_ID` (Bedrock console → Model access).
- An IAM identity (ideally an instance role, not long-lived keys) with least-privilege permissions for `s3:PutObject`/`GetObject`/`DeleteObject` on the bucket, `transcribe:StartTranscriptionJob`/`GetTranscriptionJob`, and `bedrock:InvokeModel` scoped to the model/inference-profile ARN.

## Deployment

Three low-cost deployment walkthroughs are included, depending on how you want to trade off cost, complexity, and cold-start delay:

- [`DEPLOYMENT.md`](DEPLOYMENT.md) — AWS EC2, an always-on small Graviton instance running as a systemd service. Simplest mental model, predictable ~$17/month (or under $1/month if you stop it between demos).
- [`DEPLOYMENT-LAMBDA.md`](DEPLOYMENT-LAMBDA.md) — AWS Lambda container image + Lambda Web Adapter. Scales to zero, realistically $0/month at demo volume, at the cost of ~10-20s cold starts.
- [`DEPLOYMENT-CLOUDRUN.md`](DEPLOYMENT-CLOUDRUN.md) — Google Cloud Run. Also scale-to-zero and realistically $0/month, with simpler container tooling than Lambda, at the cost of a second cloud account and storing AWS credentials as a secret instead of using an instance role.
