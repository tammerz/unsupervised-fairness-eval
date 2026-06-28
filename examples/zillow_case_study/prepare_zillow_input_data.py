# This script prepares the Zillow Prize dataset for fairness analysis.
# It loads raw data, merges training and property information, cleans the data,
# handles missing values with a focus on maintaining data integrity for fairness audits,
# and enriches the data with spatial information.

import pandas as pd
import os
import numpy as np
import geopandas as gpd
import argparse

# --- Argument Parsing ---
# Set up command-line arguments to allow for flexible filtering of the dataset.
# This is useful for focusing the analysis on specific geographic regions like a county, city, or zip code.
parser = argparse.ArgumentParser(description="Prepare Zillow input data with geographic filters.")
parser.add_argument("--fips", type=int, default=None, help="Filter by FIPS code (e.g., 6037 for LA County)")
parser.add_argument("--city_id", type=int, default=None, help="Filter by regionidcity (e.g., 26964 for Simi Valley)")
parser.add_argument("--zip_id", type=int, default=None, help="Filter by regionidzip")
parser.add_argument("--output_csv", type=str, default="Data/prepared_zillow_data.csv", help="Output path for the prepared data")
args = parser.parse_args()

# --- 1. Data Loading ---
# Load the raw Kaggle datasets for Zillow's Home Value Prediction competition.
print("Loading data...")

# Specify columns to load to reduce memory consumption and improve I/O efficiency.
train_cols = ['parcelid', 'logerror', 'transactiondate']
# Define data types for property features to optimize memory usage.
prop_dtypes = {
    'parcelid': 'Int32',
    'regionidzip': 'Int32',
    'regionidneighborhood': 'Int32',
    'regionidcity': 'Int32',
    'latitude': 'float32',
    'longitude': 'float32',
    'propertylandusetypeid': 'float32',
    'calculatedfinishedsquarefeet': 'float32',
    'yearbuilt': 'float32',
    'bathroomcnt': 'float32',
    'bedroomcnt': 'float32',
    'lotsizesquarefeet': 'float32',
    'garagetotalsqft': 'float32',
    'roomcnt': 'float32',
    'rawcensustractandblock': 'float64',
    'fips': 'Int32',
    'taxvaluedollarcnt': 'float32',
    'buildingqualitytypeid': 'float32',
    'propertyzoningdesc': 'string',
    'taxdelinquencyflag': 'string'
}

# Read the training data (transaction records) and property data.
train_df = pd.read_csv('Data/zillow_input_data/train_2016_v2.csv', usecols=train_cols)
properties_df = pd.read_csv('Data/zillow_input_data/properties_2016.csv', usecols=list(prop_dtypes.keys()), dtype=prop_dtypes)

# Store original shapes for comparison at the end.
orig_train_shape = train_df.shape
orig_prop_shape = properties_df.shape

# --- Data Cleaning (Pre-Merge) ---
# Clean the training data before merging to avoid unnecessary processing on irrelevant records.
print("Cleaning training data...")
# Drop records with no 'logerror', as they cannot be used for training or evaluation.
train_df = train_df.dropna(subset=['logerror'])
# Some properties were sold multiple times. We keep only the most recent transaction for each property.
train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])
train_df = train_df.sort_values('transactiondate').drop_duplicates(subset=['parcelid'], keep='last')
# The transaction date is no longer needed after this step.
train_df = train_df.drop(columns=['transactiondate'])

# --- 2. Merge Datasets ---
# Combine the transaction data (with 'logerror') and the property features.
print("Merging datasets...")
# An inner join ensures that we only work with properties that have both transaction and feature data.
clean_df = pd.merge(train_df, properties_df, on='parcelid', how='inner')

# Exit if the merge results in an empty DataFrame, which indicates a data mismatch.
if clean_df.empty:
    print("Error: No overlapping records found between training and properties data.")
    exit(1)

# Filter for Single-Family Residential properties (land use type ID 261).
# This ensures a more consistent dataset for analysis, avoiding biases from comparing
# different property types like commercial buildings or vacant land.
clean_df = clean_df[clean_df['propertylandusetypeid'] == 261]

# --- Geographic Filtering ---
# Apply filters based on the command-line arguments provided by the user.
if args.fips is not None:
    source_fips_count = (properties_df['fips'] == args.fips).sum()
    print(f"Found {source_fips_count} properties with FIPS {args.fips} in the source properties file.")
    print(f"Filtering for FIPS code {args.fips}...")
    clean_df = clean_df[clean_df['fips'] == args.fips]
    if clean_df.empty:
        print(f"Error: No properties found for FIPS {args.fips}.")
        exit(1)

if args.city_id is not None:
    print(f"Filtering for City ID {args.city_id}...")
    clean_df = clean_df[clean_df['regionidcity'] == args.city_id]
    if clean_df.empty:
        print(f"Error: No properties found for City ID {args.city_id}.")
        exit(1)

if args.zip_id is not None:
    print(f"Filtering for Zip ID {args.zip_id}...")
    clean_df = clean_df[clean_df['regionidzip'] == args.zip_id]
    if clean_df.empty:
        print(f"Error: No properties found for Zip ID {args.zip_id}.")
        exit(1)

# FAIRNESS NOTE: We intentionally do not filter out extreme 'logerror' outliers.
# In algorithmic auditing, these extreme errors can be indicative of systemic biases
# affecting certain populations. Removing them could mask these important issues.

# --- 3. Column Cleanup ---
# Drop the 'propertylandusetypeid' column as it's now redundant after filtering.
clean_df = clean_df.drop(columns=['propertylandusetypeid'])

# --- 4. Missing Value Imputation ---
print("Handling missing values...")

# The latitude and longitude are stored as scaled integers; convert them to standard decimal degrees.
clean_df.loc[:, ['latitude', 'longitude']] /= 1e6

# Drop properties with missing coordinates, as their location cannot be imputed reliably.
clean_df = clean_df.dropna(subset=['latitude', 'longitude'])

# Based on domain knowledge, missing garage square footage usually means there is no garage.
# Fill these missing values with 0.
clean_df['garagetotalsqft'] = clean_df['garagetotalsqft'].fillna(0)

# Define the structural features of the properties that we will impute.
structural_cols = [
    'calculatedfinishedsquarefeet', 'yearbuilt', 'bathroomcnt', 'bedroomcnt',
    'lotsizesquarefeet', 'garagetotalsqft', 'roomcnt'
]

# Address a known anomaly in the Zillow dataset where some properties have 0 for room/bed/bath counts.
# We replace these 0s with NaN to prepare them for imputation.
zero_inflation_cols = ['roomcnt', 'bedroomcnt', 'bathroomcnt']
for col in zero_inflation_cols:
    clean_df.loc[clean_df[col] == 0, col] = np.nan

# Perform contextual imputation: use the median value of the property's zip code to fill missing values.
# This is more accurate than using a global median.
print("Performing localized median imputation...")
local_medians = clean_df.groupby('regionidzip')[structural_cols].transform('median')
clean_df[structural_cols] = clean_df[structural_cols].fillna(local_medians)

# If a feature is still missing (e.g., an entire zip code lacks data), use the global median as a fallback.
clean_df[structural_cols] = clean_df[structural_cols].fillna(clean_df[structural_cols].median())

# As a final fallback, if a feature is entirely missing from the dataset, fill with 0.
for col in structural_cols:
    if clean_df[col].isna().all():
        clean_df[col] = clean_df[col].fillna(0)

# Address a data anomaly specific to LA County where 'roomcnt' is often omitted.
# We enforce a logical minimum for room count: at least the number of bedrooms + bathrooms + 1 (for a living area).
logical_min_rooms = clean_df['bedroomcnt'] + clean_df['bathroomcnt'] + 1
clean_df['roomcnt'] = np.maximum(clean_df['roomcnt'], logical_min_rooms)

# Round 'yearbuilt' to the nearest integer after imputation.
if 'yearbuilt' in clean_df.columns:
    clean_df['yearbuilt'] = clean_df['yearbuilt'].round().astype(int)

# --- 5. Outlier Handling and Feature Engineering ---
# FAIRNESS NOTE: Structural and proxy feature outliers are preserved.
# Algorithms like K-Medoids are robust to these outliers. Removing them could erase
# genuine socioeconomic disparities between neighborhoods, which is critical for fairness analysis.

# Format geographic and proxy IDs as string type for categorical treatment, filling NaNs with -1.
clean_df['regionidzip'] = clean_df['regionidzip'].fillna(-1).astype('string')
clean_df['regionidneighborhood'] = clean_df['regionidneighborhood'].fillna(-1).astype('string')
clean_df['regionidcity'] = clean_df['regionidcity'].fillna(-1).astype('string')
clean_df['fips'] = clean_df['fips'].fillna(-1).astype('string')
clean_df['rawcensustractandblock'] = clean_df['rawcensustractandblock'].fillna(-1).astype('string')

# Clean up categorical proxy features.
clean_df['taxdelinquencyflag'] = clean_df['taxdelinquencyflag'].fillna('N').astype('string')
clean_df['propertyzoningdesc'] = clean_df['propertyzoningdesc'].fillna('Unknown').astype('string')

# Impute continuous financial/quality proxies using localized medians to avoid distorting wealth data.
proxy_continuous = ['taxvaluedollarcnt', 'buildingqualitytypeid']
local_proxy_medians = clean_df.groupby('regionidzip')[proxy_continuous].transform('median')
clean_df[proxy_continuous] = clean_df[proxy_continuous].fillna(local_proxy_medians)
clean_df[proxy_continuous] = clean_df[proxy_continuous].fillna(clean_df[proxy_continuous].median())

# Final fallback for proxy features.
for col in proxy_continuous:
    if clean_df[col].isna().all():
        clean_df[col] = clean_df[col].fillna(0)

# FAIRNESS NOTE: Bin continuous wealth proxies (like tax value) into categorical groups (quartiles).
# This is necessary for certain fairness tests (e.g., Chi-Squared) that require discrete groups.
n_bins = pd.qcut(clean_df['taxvaluedollarcnt'], q=4, duplicates='drop').nunique()
safe_labels = ['Q1_LowWealth', 'Q2_MedWealth', 'Q3_MedHighWealth', 'Q4_HighWealth'][:n_bins]
clean_df['taxvalue_quartile'] = pd.qcut(clean_df['taxvaluedollarcnt'], q=4, labels=safe_labels, duplicates='drop').astype('string')

# --- 6. Spatial Data Conversion ---
# Convert the pandas DataFrame into a GeoDataFrame to enable spatial operations.
print("Converting to Spatial Data (GeoDataFrame)...")
clean_df = gpd.GeoDataFrame(
    clean_df,
    geometry=gpd.points_from_xy(clean_df['longitude'], clean_df['latitude']),
    crs="EPSG:4326"  # Set the coordinate reference system to WGS 84.
)

# Convert the spatial geometry column to Well-Known Text (WKT) format,
# which is a string representation that can be saved in a CSV file.
wkt_geometries = clean_df['geometry'].to_wkt()
clean_df = pd.DataFrame(clean_df) # Convert back to a standard DataFrame for saving.
clean_df['geometry'] = wkt_geometries

# --- 7. Export Data ---
# Save the prepared data to a CSV file for use in subsequent analysis.
output_path = args.output_csv
out_dir = os.path.dirname(output_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True) # Create the output directory if it doesn't exist.
clean_df.to_csv(output_path, index=False)

# Print summary information about the data preparation process.
print(f"Original train data shape: {orig_train_shape}")
print(f"Original properties data shape: {orig_prop_shape}")
print(f"Data prepared and saved to {output_path}. Final shape: {clean_df.shape}")
