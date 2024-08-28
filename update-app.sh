#!/bin/bash

# Set variables
PEM_FILE="/Users/dunstanmatekenya/Google Drive/My Drive/AWS-Utils/nyasa.pem"
IP_ADDRESS_AND_USER="ubuntu@ec2-3-220-232-45.compute-1.amazonaws.com"
REPO_DIR="/home/ubuntu/Phoso-AI/"

# ====================================
# PUSH UPDATED CODE TO GITHUB
# ====================================
git add .
git commit -m "Updated code"
git push origin main

# ====================================
# CONNECT TO INSTANCE AND UPDATE APP
# ====================================
# Connect to instance
ssh -i "$PEM_FILE" $IP_ADDRESS_AND_USER << EOF

# Navigate to the application directory
cd $REPO_DIR

# Pull updates from GitHub
git pull origin main

# Restart the FastAPI service
sudo systemctl restart fastapi

# Optionally, check the status of the service
sudo systemctl status fastapi
EOF

echo "Application has been successfully updated and restarted on the EC2 instance."
