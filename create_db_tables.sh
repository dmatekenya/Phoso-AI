#!/bin/bash
DB_NAME="food_security"
DB_USER="dunstanmatekenya"
DB_PASS="Khama2012"
DB_HOST="localhost"
DB_PORT="5432"
SQL_FILE="create_psql_tables.sql"

# =====================================
# CREATE DATABASE IF DOESNT EXIST
# ======================================
export PGPASSWORD=$DB_PASS

# Check if the database exists and create it if it doesnt
if psql -h $DB_HOST -p $DB_PORT -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "Database $DB_NAME exists. Dropping and recreating it..."
    dropdb -h $DB_HOST -p $DB_PORT -U "$DB_USER" "$DB_NAME"
else
    echo "Database $DB_NAME does not exist. Creating it..."
fi

# Create the new database
createdb -h $DB_HOST -p $DB_PORT -U "$DB_USER" "$DB_NAME"

# =====================================
# ADD TABLES TO THE DATABASE
# =====================================
# Run the SQL script
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $SQL_FILE

# Unset the password variable for security reasons
unset PGPASSWORD
