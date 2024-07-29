/*
CREATE MEDATA TABLES
*/
-- Table to hold table metadata
CREATE TABLE table_metadata (
    table_name VARCHAR(255) PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to hold column metadata
CREATE TABLE column_metadata (
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (table_name, column_name)
);

/*
CREATE TABLES FOR COMMODITY PRICES
*/
-- Create table 24.1.1
CREATE TABLE commodity_prices (
    ADD_Name VARCHAR(25),
    EPA_name VARCHAR(25),
    District VARCHAR(25),
    Market VARCHAR(25),
    Month_Name VARCHAR(25),
    Yr NUMERIC,
    Collection_Date DATE,
    Commodity VARCHAR(25),
    Price NUMERIC
);

-- Load data into table from CSV file
-- Update and change path to CSV file accordingly 
\COPY commodity_prices(ADD_Name,EPA_Name,District,Market,Month_Name,Yr,Commodity,Price,Collection_Date) FROM '/Users/dunstanmatekenya/Library/CloudStorage/GoogleDrive-dmatekenya@gmail.com/My Drive/Chichewa-ASR/phosoAI-whatsapp-app/data/tables/prices.csv' DELIMITER ',' CSV HEADER;

-- Add metadata for this table 
INSERT INTO table_metadata (table_name, description)
VALUES ('commodity_prices', 'Prices in Malawi Kwacha per Kg for various food commodities for 2024 in Malawi');

-- Add metadata for columns in this table 
INSERT INTO column_metadata (table_name, column_name, description)
VALUES
    ('commodity_prices', 'ADD_Name', 'Name identifying the ADD.'),
    ('commodity_prices', 'EPA', 'Name of the EPA.'),
    ('commodity_prices', 'District', 'This is the name of the district in Malawi where the commodity is being sold.'),
    ('commodity_prices', 'Market', 'This is the name of the market within the district where the commodity is being sold'),
    ('commodity_prices', 'Month_Name', 'This is the month of the year.'),
    ('commodity_prices', 'Yr', 'This is the Year.'),
    ('commodity_prices', 'Collection_Date', 'Data collection date'),
    ('commodity_prices', 'Commodity', 'This is the name of the food commodity (e.g., Maize)'),
    ('commodity_prices', 'Price', 'This is the average price of the commodity. This price is in Kwach per Kg')
    ;


-- Create table production
CREATE TABLE production (
    District VARCHAR(25),
    Crop VARCHAR(25),
    Yield NUMERIC,
    Season VARCHAR(25)
);

-- Load data into table from CSV file
-- Update and change path to CSV file accordingly 
\COPY production(District,Crop,Yield,Season) FROM '/Users/dunstanmatekenya/Library/CloudStorage/GoogleDrive-dmatekenya@gmail.com/My Drive/Chichewa-ASR/phosoAI-whatsapp-app/data/tables/production.csv' DELIMITER ',' CSV HEADER;


-- Add metadata for this table 
INSERT INTO table_metadata (table_name, description)
VALUES ('production', 'This tables provides information about crop production (yield) harvested in metric tonnes for the 2023-2024 growing season disagreggated at district level in Malawi');

-- Add metadata for columns in this table 
INSERT INTO column_metadata (table_name, column_name, description)
VALUES
    ('production', 'District', 'Name of the district in Malawi.'),
    ('production', 'Crop', 'Name of the crop (e.g., soy beans).'),
    ('production', 'Yield', 'Yields harvested in tonnes. '),
    ('production', 'Season', 'This the growing season when the crops were harvested.')
    ;

-- Create table production
CREATE TABLE food_insecurity (
    district VARCHAR(25),
    analyzed_population NUMERIC,
    time_period VARCHAR(25),
    insecurity_level NUMERIC,
    percentage_population NUMERIC,
    insecurity_desc_short VARCHAR(50),
    insecurity_desc_long VARCHAR(1000)
);

-- Load data into table from CSV file
-- Update and change path to CSV file accordingly 
\COPY food_insecurity(district,analyzed_population,time_period,percentage_population,insecurity_level,insecurity_desc_short,insecurity_desc_long) FROM '/Users/dunstanmatekenya/Library/CloudStorage/GoogleDrive-dmatekenya@gmail.com/My Drive/Chichewa-ASR/phosoAI-whatsapp-app/data/tables/food-insecurity.csv' DELIMITER ',' CSV HEADER;


-- Add metadata for this table 
INSERT INTO table_metadata (table_name, description)
VALUES ('food_insecurity', 'This tables provides information about food insecurity or situation about lack of food in Malawi for the period between May 2024 and March 2025.');

-- Add metadata for columns in this table 
INSERT INTO column_metadata (table_name, column_name, description)
VALUES
    ('food_insecurity', 'district', 'Name of the district in Malawi.'),
    ('food_insecurity', 'analyzed_population', 'Total population for this district'),
    ('food_insecurity', 'time_period', 'The time period being referred to for this analysis.'),
    ('food_insecurity', 'insecurity_level', 'The severity of food insecurity in increasing severity from 1 to 5.'),
    ('food_insecurity', 'percentage_population', 'Percentage of the total population who are food insecure.'),
    ('food_insecurity', 'insecurity_desc_short', 'Provides a short description of the food insecurity level. For example, 1 means households have zero or minimal food insecurity while 5 means households lack food.'),
    ('food_insecurity', 'insecurity_desc_long', 'Provides very detailed description of food insecurity.')
    ;
