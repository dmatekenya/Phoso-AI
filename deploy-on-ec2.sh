#!/bin/bash

# =========================
# PUSH CHANGES TO GITHUB
# =========================
git add .
git commit -m "Updated code"
git push origin main


# =====================================
# TRANSFER UPDATED FILES/FOLDERS TO S3
# =====================================
# Data folder which isnt available on GitHub
aws s3 cp --recursive ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/data s3://chichewa-ai/phoso-ai-files/data --profile dmatekenya

# Vector Database folder which isnt available on GitHub
aws s3 cp --recursive ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/store s3://chichewa-ai/phoso-ai-files/store --profile dmatekenya

# .env file 
aws s3 cp ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/.env s3://chichewa-ai/phoso-ai-files/ --profile dmatekenya

# Script for configuring EC2  instance  
aws s3 cp ~/Google\ Drive/My\ Drive/phosoAI-whatsapp-app-aws/ec2-setup.sh s3://chichewa-ai/phoso-ai-files/ --profile dmatekenya

# =========================
# CREATE SPOT INSTANCE
# =========================
# Create instance 
aws ec2 run-instances --cli-input-json file://ec2-instance-config2.json --profile dmatekenya

# ==================================
# CONNECT TO INSTANCE AND RUN SETUP 
# SCRIPT
# ==================================






