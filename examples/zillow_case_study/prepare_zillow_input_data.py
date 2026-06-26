import pandas as pd
import os
import numpy as np
import geopandas as gpd
import argparse

parser = argparse.ArgumentParser(description="Prepare Zillow input data with geographic filters.")
parser.add_argument("--fips", type=int, default=None, help="Filter by FIPS code (e.g., 6037 for LA County)")
parser.add_argument("--city_id", type=int, default=None, help="Filter by regionidcity (e.g., 26964 for Simi Valley)")
parser.add_argument("--zip_id", type=int, default=None, help="Filter by regionidzip")
parser.add_argument("--output_csv", type=str, default="Data/prepared_zillow_data.csv", help="Output path for the prepared data")
args = parser.parse_args()

# 1. Load the raw Kaggle datasets
print("Loading data...")

# Use 'usecols' to drastically reduce memory consumption and I/O time during loading
train_cols = ['parcelid', 'logerror', 'transactiondate']
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

train_df = pd.read_csv('Data/zillow_input_data/train_2016_v2.csv', usecols=train_cols)
properties_df = pd.read_csv('Data/zillow_input_data/properties_2016.csv', usecols=list(prop_dtypes.keys()), dtype=prop_dtypes)

orig_train_shape = train_df.shape
orig_prop_shape = properties_df.shape

# Pre-merge pruning: drop missing targets and duplicates BEFORE the expensive join
print("Cleaning training data...")
train_df = train_df.dropna(subset=['logerror'])
# Properties sold multiple times appear multiple times; sort by date to guarantee we keep the latest transaction
train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])
train_df = train_df.sort_values('transactiondate').drop_duplicates(subset=['parcelid'], keep='last')
train_df = train_df.drop(columns=['transactiondate'])  # Drop to keep feature space clean

# 2. Merge the error metrics with the structural features
print("Merging datasets...")
# Use an inner join to ensure we only keep records that exist in both datasets
clean_df = pd.merge(train_df, properties_df, on='parcelid', how='inner')

if clean_df.empty:
    print("Error: No overlapping records found between training and properties data.")
    exit(1)

# Filter for Single-Family Residential properties (Code 261)
# Ensures apples-to-apples fairness analysis and prevents our zero-inflation logic 
# from erroneously imputing rooms onto vacant lots or commercial properties.
clean_df = clean_df[clean_df['propertylandusetypeid'] == 261]

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

# FAIRNESS UPDATE: We intentionally do NOT filter out extreme logerror outliers.
# In algorithmic auditing, extreme predictive errors often represent the very 
# systemic biases or harms impacting marginalized populations. Dropping them 
# artificially sanitizes the algorithm's performance.

# 3. Clean up columns
# Keep parcelid, drop only propertylandusetypeid
clean_df = clean_df.drop(columns=['propertylandusetypeid'])

# 4. Handle Missing Values (Mitigating Survivorship Bias)
print("Handling missing values...")

# Convert scaled integer coordinates back to standard decimal degrees for interpretability
clean_df.loc[:, ['latitude', 'longitude']] /= 1e6

# Drop properties with missing coordinates BEFORE imputation
# We cannot guess a location without creating artificial density clumps that break K-Medoids
clean_df = clean_df.dropna(subset=['latitude', 'longitude'])

# Domain Knowledge: Missing garage square footage typically indicates NO garage.
# We explicitly fill this with 0 before local median imputation so we don't conjure garages out of thin air.
clean_df['garagetotalsqft'] = clean_df['garagetotalsqft'].fillna(0)

# Define features for imputation
structural_cols = [
    'calculatedfinishedsquarefeet', 'yearbuilt', 'bathroomcnt', 'bedroomcnt', 
    'lotsizesquarefeet', 'garagetotalsqft', 'roomcnt'
]

# Fix known Zillow dataset anomaly: zero-inflation in structural counts
# Many homes incorrectly list 0 rooms, beds, or baths despite being finished structures
zero_inflation_cols = ['roomcnt', 'bedroomcnt', 'bathroomcnt']
for col in zero_inflation_cols:
    clean_df.loc[clean_df[col] == 0, col] = np.nan

# Contextual Imputation: Fill missing features using the local median of the property's Zip Code
print("Performing localized median imputation...")
local_medians = clean_df.groupby('regionidzip')[structural_cols].transform('median')
clean_df[structural_cols] = clean_df[structural_cols].fillna(local_medians)

# Fallback: If an entire Zip Code is missing a feature, use the global median
clean_df[structural_cols] = clean_df[structural_cols].fillna(clean_df[structural_cols].median())

# Final fallback if the feature is completely missing in the target region
for col in structural_cols:
    if clean_df[col].isna().all():
        clean_df[col] = clean_df[col].fillna(0)

# Domain Knowledge: California County Reporting Anomaly (LA County omits roomcnt)
# The global median fallback can assign a 5-bed mansion and a 1-bed shack the exact same roomcnt.
# Enforce a logical minimum: A home must have at least (beds + baths + 1 for living area).
logical_min_rooms = clean_df['bedroomcnt'] + clean_df['bathroomcnt'] + 1
clean_df['roomcnt'] = np.maximum(clean_df['roomcnt'], logical_min_rooms)

# Final cleanup: Round yearbuilt to nearest integer after imputation
if 'yearbuilt' in clean_df.columns:
    clean_df['yearbuilt'] = clean_df['yearbuilt'].round().astype(int)

# 5. Handle Extreme Outliers 
# FAIRNESS UPDATE: Structural and proxy outliers are preserved. 
# K-Medoids (using Gower distance on coordinates) is robust to structural outliers.
# Furthermore, clipping structural/financial bounds (like tiny lot sizes or massive 
# tax valuations) erases genuine socioeconomic disparities between neighborhoods.

# Format Geographic and Proxy IDs: Fill missing with -1 and convert to string for categorical treatment
clean_df['regionidzip'] = clean_df['regionidzip'].fillna(-1).astype('string')
clean_df['regionidneighborhood'] = clean_df['regionidneighborhood'].fillna(-1).astype('string')
clean_df['regionidcity'] = clean_df['regionidcity'].fillna(-1).astype('string')
clean_df['fips'] = clean_df['fips'].fillna(-1).astype('string')
clean_df['rawcensustractandblock'] = clean_df['rawcensustractandblock'].fillna(-1).astype('string')

# Clean up text/categorical proxies
clean_df['taxdelinquencyflag'] = clean_df['taxdelinquencyflag'].fillna('N').astype('string')
clean_df['propertyzoningdesc'] = clean_df['propertyzoningdesc'].fillna('Unknown').astype('string')

# Median impute continuous financial/quality proxies
# FAIRNESS UPDATE: Use localized medians to prevent artificially inflating wealth in under-resourced areas.
proxy_continuous = ['taxvaluedollarcnt', 'buildingqualitytypeid']
local_proxy_medians = clean_df.groupby('regionidzip')[proxy_continuous].transform('median')
clean_df[proxy_continuous] = clean_df[proxy_continuous].fillna(local_proxy_medians)
clean_df[proxy_continuous] = clean_df[proxy_continuous].fillna(clean_df[proxy_continuous].median())

# Final fallback if the proxy is completely missing in the target region
for col in proxy_continuous:
    if clean_df[col].isna().all():
        clean_df[col] = clean_df[col].fillna(0)

# FAIRNESS UPDATE: Bin continuous wealth proxies into categorical groups.
# Algorithmic fairness tests (like Chi-Squared and Poisson) require distinct groups.
# Testing tens of thousands of unique dollar amounts causes massive memory array crashes.
# Generate safe labels in case duplicates='drop' merges quartile boundaries
n_bins = pd.qcut(clean_df['taxvaluedollarcnt'], q=4, duplicates='drop').nunique()
safe_labels = ['Q1_LowWealth', 'Q2_MedWealth', 'Q3_MedHighWealth', 'Q4_HighWealth'][:n_bins]
clean_df['taxvalue_quartile'] = pd.qcut(clean_df['taxvaluedollarcnt'], q=4, labels=safe_labels, duplicates='drop').astype('string')

# 6. Convert Zillow Data to Spatial Data
print("Converting to Spatial Data (GeoDataFrame)...")
clean_df = gpd.GeoDataFrame(
    clean_df,
    geometry=gpd.points_from_xy(clean_df['longitude'], clean_df['latitude']),
    crs="EPSG:4326"
)

# Explicitly convert spatial geometry to Well-Known Text (WKT) string before saving
wkt_geometries = clean_df['geometry'].to_wkt()
clean_df = pd.DataFrame(clean_df)
clean_df['geometry'] = wkt_geometries

# 7. Export the data for the c4f tool
output_path = args.output_csv
out_dir = os.path.dirname(output_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
clean_df.to_csv(output_path, index=False)
print(f"Original train data shape: {orig_train_shape}")
print(f"Original properties data shape: {orig_prop_shape}")
print(f"Data prepared and saved to {output_path}. Final shape: {clean_df.shape}")