#!/bin/bash

# =====================================
# TRANSFER UPDATED FILES/FOLDERS TO S3
# =====================================
aws s3 cp --recursive ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/data s3://chichewa-ai/phoso-ai-files/data --profile dmatekenya
aws s3 cp --recursive ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/store s3://chichewa-ai/phoso-ai-files/store --profile dmatekenya
aws s3 cp ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/.env s3://chichewa-ai/phoso-ai-files/ --profile dmatekenya
aws s3 cp ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/ec2-setup.sh s3://chichewa-ai/phoso-ai-files/ --profile dmatekenya

# =========================
# CREATE SPOT INSTANCE
# =========================
# Create instance 
# aws ec2 run-instances --cli-input-json file://ec2-instance-config.json --profile dmatekenya

# =========================
# PUSH CHANGES TO GITHUB
# =========================
git add .
git commit -m "Updated code"
git push origin main

# ==================================
# CONNECT TO INSTANCE AND RUN SETUP 
# SCRIPT
# ==================================







