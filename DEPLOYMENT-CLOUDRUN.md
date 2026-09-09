# Deploying to Google Cloud Run

**Deployment strategy: Google Cloud Run, container, scale-to-zero.** Other strategies: [`DEPLOYMENT.md`](DEPLOYMENT.md) (EC2, always-on VM) | [`DEPLOYMENT-LAMBDA.md`](DEPLOYMENT-LAMBDA.md) (AWS Lambda container)

## Why this one

Cloud Run's always-free monthly allocation is generous — 2 million requests, 240,000 vCPU-seconds, and 450,000 GiB-seconds — and it scales to zero exactly like Lambda, so idle time costs nothing. Unlike the Lambda path, `app.py` runs as a completely ordinary Docker container here: no adapter layer, no read-only filesystem restrictions, just `python3 app.py` inside a normal container image. The trade-off is that this app still needs to call Amazon S3, Transcribe, and Bedrock, and Cloud Run has no equivalent of an EC2/Lambda instance role for AWS — you'll authenticate to AWS with an IAM user's access keys stored as a Cloud Run secret instead. You'll also need a Google Cloud account and project with billing enabled (required to use Cloud Run at all, even fully within the free tier — you won't be charged unless you exceed the free allocation).

The same caveat about `app.py`'s blocking Transcribe polling loop applies here as on Lambda: a long recording holds the container active (and billed) for the whole wait. Fine for occasional demo use; see `DEPLOYMENT-LAMBDA.md` for the full explanation if you want the details.

## 1. Enable Bedrock model access (in AWS)

Bedrock console -> **Model access** -> request/enable the Claude model your `BEDROCK_MODEL_ID` points to, in your target AWS region. This is unrelated to Google Cloud and needs doing regardless of where the app itself runs.

## 2. Create a scoped AWS IAM user for cross-cloud access

Since Cloud Run can't assume an AWS role, create a dedicated IAM user in AWS console -> IAM -> Users -> Create user (no console access needed, just programmatic access), and attach this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::telehealth-capstone-data-alm",
        "arn:aws:s3:::telehealth-capstone-data-alm/*"
      ]
    },
    {
      "Sid": "TranscribeAccess",
      "Effect": "Allow",
      "Action": ["transcribe:StartTranscriptionJob", "transcribe:GetTranscriptionJob"],
      "Resource": "*"
    },
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:*:YOUR_ACCOUNT_ID:inference-profile/us.anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
      ]
    }
  ]
}
```

Generate an access key for this user (Security credentials tab -> Create access key). Keep the secret key somewhere safe for step 6 — never commit it to the repo.

## 3. Install and authenticate the gcloud CLI

```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

## 4. Write the Dockerfile

Add this `Dockerfile` to the repo root, next to `app.py` (no Lambda adapter needed here):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the sentiment model into the image at build time so cold starts don't
# re-download it from Hugging Face on every scale-from-zero.
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN python3 -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"

COPY app.py index.html ./

ENV PORT=8080
EXPOSE 8080

CMD ["python3", "app.py"]
```

Cloud Run injects `PORT=8080` by default and expects the container to listen on it — `app.py` already reads `PORT` from the environment and binds `0.0.0.0`, so this works without further changes.

## 5. Store the AWS secret key in Secret Manager

```bash
printf '%s' 'YOUR_AWS_SECRET_ACCESS_KEY' | gcloud secrets create aws-secret-access-key --data-file=-
```

You'll grant Cloud Run access to this secret when you deploy in the next step.

## 6. Deploy

`gcloud run deploy` builds the container (via Cloud Build) and deploys it in one step — no manual image push needed:

```bash
gcloud run deploy telehealth-screener \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --min-instances 0 \
  --set-env-vars S3_BUCKET=telehealth-capstone-data-alm,AWS_REGION=us-east-1,BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6,AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID \
  --set-secrets AWS_SECRET_ACCESS_KEY=aws-secret-access-key:latest
```

`--allow-unauthenticated` makes it a public demo, matching the other two strategies. `--min-instances 0` keeps it scaled to zero (and free) when idle — setting this to `1` would eliminate cold starts but keeps a container warm continuously, which bills like an always-on server and defeats the point of this approach.

## 7. Test it

The deploy command prints a service URL (something like `https://telehealth-screener-xxxxx-uc.a.run.app`) — open it in a browser. The first request after a period of no traffic will be slower while the container starts and the model loads from the baked-in cache.

## Cost estimate

At demo-level usage, this comfortably fits inside the always-free 2M requests / 240,000 vCPU-second / 450,000 GiB-second monthly allowance — realistically **$0/month**. The only ongoing cost is a few cents at most for storing the built container image in Artifact Registry. If you've never used a Google Cloud free trial before, the console may also offer a one-time introductory credit on top of this — worth checking, but not something to plan around since terms change.

## Locking it down later

For a portfolio demo, `--allow-unauthenticated` is usually what you want. If you'd rather it not be publicly invokable, redeploy with `--no-allow-unauthenticated` and grant the `roles/run.invoker` IAM role only to specific accounts.
