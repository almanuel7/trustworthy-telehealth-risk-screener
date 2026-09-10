# Deploying to AWS (low-cost EC2)

**Deployment strategy: AWS EC2 (always-on VM).** Other low cost strategy: [`DEPLOYMENT-CLOUDRUN.md`](DEPLOYMENT-CLOUDRUN.md) (Google Cloud Run)

This app needs a small always-running server (it keeps a PyTorch sentiment model loaded in memory), so EC2 is a better fit than serverless here. This walkthrough uses the cheapest instance size that reliably avoids out-of-memory crashes, plus a systemd service so it restarts itself if it ever crashes or the instance reboots.

## 1. Enable Bedrock model access

Bedrock console -> **Model access** (left sidebar) -> request/enable access to the Claude model your `BEDROCK_MODEL_ID` points to, in the region you'll deploy to (us-east-1 by default). This can take a few minutes to show "Access granted."

## 2. Create an IAM role for the instance (no long-lived keys)

IAM console -> Roles -> Create role -> Trusted entity: **AWS service** -> Use case: **EC2** -> attach a new inline/custom policy:

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

Replace `YOUR_ACCOUNT_ID` with your 12-digit AWS account ID, and the bucket name if you use a different one. Transcribe doesn't support per-resource restriction, so `"*"` there is normal. The Bedrock statement grants both the inference profile and the underlying foundation models, because cross-region inference profiles (the `us.` prefix) route requests through several regions under the hood.

Name the role something like `telehealth-screener-ec2-role`.

## 3. Launch the EC2 instance

- AMI: **Amazon Linux 2023 (arm64)**
- Instance type: **t4g.small** (2 vCPU / 2 GiB RAM, Graviton/ARM). This is the cheapest size with enough headroom for PyTorch + the RoBERTa model without OOM crashes — t4g.micro's 1 GiB is too tight.
- Key pair: create or select one for SSH.
- Network: default VPC, auto-assign public IP = enabled.
- IAM role: attach the role from step 2.
- Storage: 10 GiB gp3 is plenty.
- Security group:
  - SSH (22) from "My IP" only
  - Custom TCP 8000 from 0.0.0.0/0 (or narrow it down once you're done testing)

## 4. Connect and install

```bash
ssh -i your-key.pem ec2-user@<public-ip>

sudo dnf update -y
sudo dnf install -y git python3 python3-pip
git clone https://github.com/<your-username>/trustworthy-telehealth-risk-screener.git
cd trustworthy-telehealth-risk-screener
pip3 install --user -r requirements.txt
```

## 5. Configure environment variables

```bash
cat > ~/telehealth.env << 'ENV'
S3_BUCKET=telehealth-capstone-data-alm
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
PORT=8000
ENV
```

No AWS access keys go here — the instance role supplies credentials automatically via boto3's default credential chain.

## 6. Run it as a systemd service (auto-restarts on crash or reboot)

```bash
sudo tee /etc/systemd/system/telehealth.service > /dev/null << 'UNIT'
[Unit]
Description=Trustworthy Telehealth Risk Screener
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/trustworthy-telehealth-risk-screener
EnvironmentFile=/home/ec2-user/telehealth.env
ExecStart=/usr/bin/python3 app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now telehealth.service
sudo systemctl status telehealth.service
```

## 7. Test it

Open `http://<public-ip>:8000` in a browser. First request after a (re)start will be slower while the sentiment model loads into memory.

## 8. Cost controls

Set a budget alert so you're never surprised: Billing console -> **Budgets** -> Create budget -> e.g. alert at $10 and $20/month, sent to your email.

### Estimated monthly cost (us-east-1, no free tier)

| Item | Cost |
|---|---|
| t4g.small compute, running 24/7 | ~$12.40/mo |
| Public IPv4 address (charged whenever allocated) | ~$3.65/mo |
| 10 GiB gp3 storage | ~$0.80/mo |
| **Total, always on** | **~$16.85/mo** |
| Bedrock (Claude) + Transcribe usage | pay-per-use, small at demo volume |

The first 100 GB/month of data transfer out is "Always Free" on every AWS account regardless of age, so bandwidth won't add cost at demo traffic levels.

### Cheaper option: stop it when you're not demoing

EC2 console -> select instance -> Instance state -> **Stop**. Compute and the public-IP charge both stop immediately; only the ~$0.80/mo storage keeps accruing while stopped. Starting it back up takes 1-2 minutes (boot + reloading the model on the first request), and the public IP will be different each time you start it unless you pay for a static Elastic IP (which costs the same ~$3.65/mo whether the instance is running or not). For occasional demo use, this drops your cost to well under a dollar a month at the expense of a short warm-up delay and a changing address.
