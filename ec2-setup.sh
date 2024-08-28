#!/bin/bash
# =============================
# PRELIMINARY SETUP 
# =============================
# Setup working directory
APP_DIR="/home/ubuntu/Phoso-AI"

# Retrieve ip-address
# Step 1: Get a token
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Step 2: Use the token to retrieve the public DNS name
IP_ADDRESS=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/public-hostname")

# Update package list and upgrade all packages
sudo apt update && sudo apt upgrade -y

# Install necessary packages
sudo apt install -y python3 python3-venv python3-pip git libpq-dev python3-dev build-essential nginx

# Clone repo 
git clone https://github.com/dmatekenya/Phoso-AI.git

# Copy data folders and .env file from S3 to working folder
aws s3 cp --recursive s3://chichewa-ai/phoso-ai-files/data $APP_DIR/data
aws s3 cp --recursive s3://chichewa-ai/phoso-ai-files/store $APP_DIR/store
aws s3 cp s3://chichewa-ai/phoso-ai-files/.env $APP_DIR


# =================================
# INSTALL AND CONFIGRE POSTGRESQL
# =================================
# Install postgre-SQL
sudo apt install -y postgresql postgresql-contrib

# Set variables
PG_VERSION=$(psql -V | awk '{print $3}' | awk -F. '{print $1}')
PG_CONF_DIR="/etc/postgresql/$PG_VERSION/main"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="Khama2012"
DB_USER="dmatekenya"
DB_HOST="localhost"
DB_PORT="5432"
DB_PASSWORD="Khama2012"
DB_MAIN="food_security"
DB_CONVERSATIONS="foodsec_chats"
SQL_FILE="/home/ubuntu/Phoso-AI/create_psql_tables.sql"

# Switch to the postgres user and set up the database using the variables
sudo -i -u $POSTGRES_USER bash << EOF
psql -c "ALTER USER $POSTGRES_USER PASSWORD '$POSTGRES_PASSWORD';"
psql -c "CREATE DATABASE $DB_MAIN;"
psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
psql -c "ALTER ROLE $DB_USER WITH SUPERUSER;"
psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_MAIN TO $DB_USER;"
psql -c "CREATE DATABASE $DB_CONVERSATIONS;"
psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_CONVERSATIONS TO $DB_USER;"
psql -c "ALTER SCHEMA public OWNER TO $DB_USER;"
EOF

# Run the SQL script to create tables in the food_security database
export PGPASSWORD=$DB_PASSWORD
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_MAIN -f $SQL_FILE
unset PGPASSWORD

# Configure PostgreSQL to allow remote connections
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" $PG_CONF_DIR/postgresql.conf
echo "host    all             all             0.0.0.0/0               md5" | sudo tee -a $PG_CONF_DIR/pg_hba.conf

# Restart PostgreSQL to apply the changes
sudo systemctl restart postgresql

# =================================
# SETIP PYTHON ENVIRONMENT
# =================================
cd $APP_DIR
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# =================================
# CONFIG FASTAPI APP
# =================================
# Create a systemd service file for FastAPI app
sudo bash -c "cat > /etc/systemd/system/fastapi.service <<EOT
[Unit]
Description=FastAPI application
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=$APP_DIR
Environment=\"PATH=$APP_DIR/.venv/bin\"
ExecStart=$APP_DIR/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
EOT"

# Reload systemd to apply changes
sudo systemctl daemon-reload

# Start and enable the FastAPI service
sudo systemctl start fastapi
sudo systemctl enable fastapi

# Set up Nginx as a reverse proxy
# Manually replace server name
# Set up Nginx as a reverse proxy
# Set up Nginx as a reverse proxy
sudo bash -c "cat > /etc/nginx/sites-available/fastapi <<'EOT'
server {
    listen 80;
    server_name $IP_ADDRESS;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOT"

# Remove the existing symbolic link if it exists
if [ -L /etc/nginx/sites-enabled/fastapi ]; then
    sudo rm /etc/nginx/sites-enabled/fastapi
fi

# Enable the Nginx configuration
sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled

# Test Nginx configuration and restart Nginx
if sudo nginx -t; then
    sudo systemctl restart nginx
else
    echo "Nginx configuration is invalid. Please check the configuration file."
    exit 1
fi