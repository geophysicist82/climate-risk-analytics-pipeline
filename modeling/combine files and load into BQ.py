# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:35:11 2026

@author: jenny
"""
# data location: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
import os
import glob
from google.cloud import bigquery
import pandas as pd

# 1. Authenticate with your downloaded JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\jenny\OneDrive\Desktop\code\NOAA weather data\datatest-498117-a06f231a1a88.json"

# Initialize BigQuery Client
project_id = "datatest-498117"
dataset_id = "clean_storm_data"
client = bigquery.Client(project=project_id)

# Path to where your unzipped NOAA files live
data_folder = r"C:\Users\jenny\OneDrive\Desktop\code\NOAA weather data\archive" 

def upload_file_type(file_pattern, destination_table):
    """Finds all files matching a pattern and appends them to a single BigQuery table"""
    search_path = os.path.join(data_folder, f"*{file_pattern}*.csv")
    all_files = sorted(glob.glob(search_path))
    
    if not all_files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(all_files)} files for {destination_table}. Combining data locally...")
    
    df_list = []
    for file_path in all_files:
        # Read everything, forcing pandas to keep raw structures intact
        df = pd.read_csv(file_path, low_memory=False)
        df_list.append(df)
        
    master_df = pd.concat(df_list, ignore_index=True)
    
    # --- CRUCIAL FIX FOR MIXED TYPES ---
    # Convert mixed numeric/string identity columns entirely to text strings 
    # so PyArrow stops guessing wrong and crashing.
    columns_to_force_string = [
        "EPISODE_ID", "EVENT_ID", "DAMAGE_PROPERTY", "DAMAGE_CROPS", 
        "CATEGORY", "WFO", "SOURCE", "BIZ_ENTERPRISE"
    ]
    
    for col in columns_to_force_string:
        if col in master_df.columns:
            master_df[col] = master_df[col].astype(str).str.strip()
            # Replace Python's 'nan' text with actual empty values
            master_df[col] = master_df[col].replace("nan", "")
    
    # Ensure MAGNITUDE is explicitly numeric (floats)
    if "MAGNITUDE" in master_df.columns:
        master_df["MAGNITUDE"] = pd.to_numeric(master_df["MAGNITUDE"], errors="coerce")
    # -----------------------------------
        
    print(f"Successfully combined. Total rows: {len(master_df)}")
    print(f"Streaming master data up to BigQuery table: {destination_table}...")
    
    table_ref = f"{project_id}.{dataset_id}.{destination_table}"
    
    # Load the data frame directly into BigQuery
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
    job = client.load_table_from_dataframe(master_df, table_ref, job_config=job_config)
    job.result() # Wait for the upload to complete
    
    print(f"Successfully loaded {destination_table} into BigQuery!\n")

# 2. Run the pipeline for each of the three core categories
if __name__ == "__main__":
    upload_file_type("details", "staging_storm_details")
    upload_file_type("fatalities", "staging_storm_fatalities")
    upload_file_type("location", "staging_storm_locations")