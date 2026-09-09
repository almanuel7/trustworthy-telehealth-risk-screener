# Deploying to AWS Lambda (container image) + Lambda Web Adapter

**Deployment strategy: AWS Lambda, container image, scale-to-zero.** Other strategies: [`DEPLOYMENT.md`](DEPLOYMENT.md) (EC2, always-on VM) | [`DEPLOYMENT-CLOUDRUN.md`](DEPLOYMENT-CLOUDRUN.md) (Google Cloud Run)

## Why this one

Lambda's free tier is permanent, not a 12-month thing: 1 million requests and 400,000 GB-seconds of compute every month, forever. Unlike the EC2/Lightsail plans, there is no charge at all while nothing is happening — no hourly instance cost, no public-IP charge. For occasional demo traffic this realistically costs **$0/month**.

The trade-off is cold starts: after a period of no requests, Lambda has to spin up a fresh container and reload PyTorch + the sentiment model before it can answer, which takes roughly 10-20 seconds. There's also an architectural wrinkle specific to this app — see **Important: the Transcribe polling loop** below — read that before you commit to this path.

This uses the [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter), which lets `app.py` run completely unmodified: no rewriting FastAPI routes for a different framework, no `Mangum` handler. The adapter runs your app as a normal HTTP server inside the container and proxies Lambda invocations to it.

## Important: the Transcribe polling loop

`app.py`'s `get_transcription_result()` blocks in a `while True` loop, sleeping 5 seconds at a time until the Transcribe job finishes. On EC2 that's fine — the server just sits there. On Lambda, you are billed for every second of that wait as compute time (GB-seconds), and the whole request is subject to Lambda's hard 15-minute maximum execution time. For short clips this is a non-issue and still comfortably free at low volume. For longer recordings, or if you ever expect real traffic rather than occasional demo use, this polling pattern will burn through the free tier fast and risks timing out entirely — the correct fix would be to make transcription asynchronous (e.g., trigger the analysis from an S3/EventBridge event instead of blocking inside the request), which is a real code change beyond this deployment guide. For a personal demo used occasionally, it's fine as-is; just don't be surprised if a long recording takes a while and costs a bit more than a short one.

## 1. Enable Bedrock model access

Same as the EC2 guide: Bedrock console -> **Model access** -> request/enable the Claude model your `BEDROCK_MODEL_ID` points to, in your target region.

## 2. Create the Lambda execution role

IAM console -> Roles -> Create role -> Trusted entity: **AWS service** -> Use case: **Lambda**. Attach the AWS managed policy `AWSLambdaBasicExecutionRole` (lets it write CloudWatch logs), plus this custom inline policy:

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

Replace `YOUR_ACCOUNT_ID` with your 12-digit account ID. Name the role something like `telehealth-screener-lambda-role`.

## 3. Write the Dockerfile

Add this `Dockerfile` to the repo root, next to `app.py`:

```dockerfile
FROM public.ecr.aws/docker/library/python:3.11-slim

# AWS Lambda Web Adapter: lets this container run as a plain HTTP server under Lambda,
# no code changes required in app.py
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the sentiment model into the image so cold starts don't re-download ~500MB
# from Hugging Face every time. Lambda's filesystem is read-only except /tmp, so this
# has to happen at build time, into a path that ships with the image.
ENV HF_HOME=/var/task/hf_cache
ENV TRANSFORMERS_CACHE=/var/task/hf_cache
RUN python3 -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"

COPY app.py index.html ./

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "app.py"]
```

`app.py` already binds `0.0.0.0` and reads `PORT` from the environment (from the earlier GitHub-readiness pass), so no code changes are needed for it to work behind the adapter.

## 4. Build and push the image to Amazon ECR

Build for `arm64` (Graviton) — Lambda charges roughly 20% less per GB-second on arm64, and PyTorch ships official `arm64` wheels, so there's no compatibility cost to it.

```bash
aws ecr create-repository --repository-name telehealth-screener --region us-east-1

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build --platform linux/arm64 -t telehealth-screener .

docker tag telehealth-screener:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/telehealth-screener:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/telehealth-screener:latest
```

## 5. Create the Lambda function

Lambda console -> Create function -> **Container image** -> select the image you just pushed.

- Architecture: `arm64`
- Memory: 3008 MB (more memory also means more CPU, which noticeably shortens cold starts — torch is CPU-hungry on import)
- Timeout: set generously, e.g. 300-900 seconds, to cover the Transcribe polling wait described above
- Execution role: the role from step 2

Set these environment variables on the function (Configuration -> Environment variables):

```
S3_BUCKET=telehealth-capstone-data-alm
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
```

No AWS access keys needed — the execution role supplies credentials automatically.

## 6. Add a Function URL

Configuration -> Function URL -> Create function URL -> Auth type: **NONE** (public, matching a public demo). Leave invoke mode as Buffered (the app doesn't stream responses). Using a Function URL instead of API Gateway avoids API Gateway's 29-second timeout ceiling and its extra cost — Function URLs are free.

## 7. Test it

Open the Function URL shown in the console. The first request will be slow (cold start + model load); subsequent requests within the next several minutes will be fast, since Lambda keeps the container warm for a while between invocations.

## Cost estimate

At low/demo volume (a handful of test runs a month), you'll use a tiny fraction of the 400,000 GB-second and 1,000,000 request permanent free allowances — this should land at **$0/month**. There's no public-IP charge (Function URLs don't need one) and no idle compute charge. The only way this stops being free is sustained real traffic or many long recordings that keep the function busy polling Transcribe for extended periods — not a concern for occasional demo use.
