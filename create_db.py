import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# 1. Connect to SQLite (This creates the 'credit_risk.db' file automatically)
conn = sqlite3.connect(os.path.join(BASE_DIR, "credit_risk.db"))

# 2. Read your 3 CSV files into pandas DataFrames
loans_df = pd.read_csv(os.path.join(DATASET_DIR, "loans.csv"))
customers_df = pd.read_csv(os.path.join(DATASET_DIR, "customers.csv"))
bureau_df = pd.read_csv(os.path.join(DATASET_DIR, "bureau_data.csv"))

# 3. Dump the dataframes into SQL tables inside the DB file
# if_exists='replace' ensures it overwrites if you run it multiple times
loans_df.to_sql("loans", conn, if_exists="replace", index=False)
customers_df.to_sql("customers", conn, if_exists="replace", index=False)
bureau_df.to_sql("bureau", conn, if_exists="replace", index=False)

print("Successfully created credit_risk.db with 3 relational tables!")

# 4. Always close the connection
conn.close()