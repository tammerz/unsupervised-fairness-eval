# This script performs a spatial join to enrich Zillow property data with census-level socioeconomic information,
# specifically the Area Deprivation Index (ADI). It combines prepared Zillow data with census block group
# shapefiles and ADI datasets to create a unified dataset ready for analysis.
#
# The key steps are:
#   1. Load prepared Zillow data containing property locations as WKT (Well-Known Text) geometries.
#   2. Convert the Zillow data into a GeoDataFrame.
#   3. Load a census block group shapefile for a specific region (e.g., California).
#   4. Align the Coordinate Reference Systems (CRS) of both datasets to ensure accurate spatial comparison.
#   5. Perform a spatial join (point-in-polygon) to map each property to its corresponding census block group.
#   6. Load the ADI dataset, which provides socioeconomic ranks for each census block group.
#   7. Merge the spatially joined data with the ADI data based on census identifiers (FIPS/GEOID).
#   8. Clean the final dataset by handling missing values and ensuring correct data types.
#   9. Export the enriched data to a CSV file, converting geometries back to WKT for compatibility.

import pandas as pd
import geopandas as gpd
import argparse
import os

# --- Argument Parsing ---
# Set up command-line arguments to specify input and output file paths.
parser = argparse.ArgumentParser(description="Spatial Join Zillow Data with Census ADI")
parser.add_argument("--input_csv", type=str, default="Data/prepared_zillow_data.csv", help="Input path from data prep")
parser.add_argument("--output_csv", type=str, default="Data/la_county_prepared.csv", help="Output path for joined data")
args = parser.parse_args()

# --- 1. Load and Prepare Zillow Data ---
# Load the CSV containing prepared Zillow data.
df = pd.read_csv(args.input_csv)
orig_input_shape = df.shape

# Convert the 'geometry' column from WKT strings back into spatial geometry objects.
# Note: Coordinates were previously scaled for compatibility.
df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])

# Initialize a GeoDataFrame from the pandas DataFrame.
# The 'crs' is set to EPSG:4326 (WGS 84), a standard for geographic coordinates.
zillow_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# --- 2. Load Census Block Group Shapefile ---
# Load the shapefile for California census block groups.
# GeoPandas automatically handles associated files (.dbf, .shx, .prj).
shapefile_path = "Data/shapefile_2015_BlockGroup_Calif/tl_2015_06_bg.shp"
census_gdf = gpd.read_file(shapefile_path)

# --- 3. Standardize Coordinate Reference Systems (CRS) ---
# Ensure both GeoDataFrames use the same CRS for accurate spatial operations.
# Shapefiles often use NAD83 (EPSG:4269), so we convert it to match the Zillow data's CRS.
census_gdf = census_gdf.to_crs(zillow_gdf.crs)

# --- 4. Execute the Spatial Join ---
# Perform a point-in-polygon spatial join. This operation identifies which census block group
# polygon contains each property point from the Zillow data.
joined_gdf = gpd.sjoin(zillow_gdf, census_gdf, how="left", predicate="intersects")

# Handle cases where a property might fall on a boundary, resulting in duplicate entries.
# We keep only the first match for each property.
joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]
joined_gdf = joined_gdf.drop(columns=['index_right'], errors='ignore')

# --- 5. Load and Merge Area Deprivation Index (ADI) Data ---
# Load the ADI dataset, which contains socioeconomic ranks for census block groups.
# 'FIPS' codes are loaded as strings to preserve any leading zeros.
adi_df = pd.read_csv('Data/adi_data_2015_calif_FIPS/CA_2015_ADI_Census Block Group_v3.1.csv', dtype={'FIPS': str})
orig_adi_shape = adi_df.shape

# Clean and standardize the census identifiers (GEOID and FIPS) for robust matching.
# This includes stripping whitespace, removing non-numeric prefixes, and padding to a uniform length.
joined_gdf['GEOID'] = joined_gdf['GEOID'].astype(str).str.strip().str.replace('G', '', regex=False)
joined_gdf['GEOID'] = joined_gdf['GEOID'].apply(lambda x: x.zfill(12) if x.lower() != 'nan' else x)

adi_df['FIPS'] = adi_df['FIPS'].astype(str).str.strip().str.replace('G', '', regex=False).str.zfill(12)

# Merge the spatially joined data with the ADI data on the standardized identifiers.
final_df = joined_gdf.merge(adi_df, left_on='GEOID', right_on='FIPS', how='left')

# --- 6. Clean the Final Dataset ---
# The ADI dataset uses non-numeric codes for suppressed data. Convert these to NaN.
final_df['ADI_STATERNK'] = pd.to_numeric(final_df['ADI_STATERNK'], errors='coerce')

# Remove records where the ADI rank is missing, as they cannot be used for analysis.
final_df = final_df.dropna(subset=['ADI_STATERNK'])

# Cast the cleaned ADI rank to an integer type.
final_df['ADI_STATERNK'] = final_df['ADI_STATERNK'].astype(int)

# --- 7. Validate and Export Results ---
# Check if the final DataFrame is empty, which would indicate a failure in the join or merge process.
if final_df.empty:
    print("ERROR: Resulting dataset is empty! No properties matched the ADI data.")
    print("Sample GEOIDs from spatial join:", joined_gdf['GEOID'].dropna().head().tolist())
    print("Sample FIPS from ADI data:", adi_df['FIPS'].head().tolist())
else:
    # Ensure the final DataFrame is a GeoDataFrame before converting geometry to WKT.
    if not isinstance(final_df, gpd.GeoDataFrame):
        final_df = gpd.GeoDataFrame(final_df, geometry='geometry')
        
    # Convert the geometry column back to WKT format for safe storage in a CSV file.
    wkt_geometries = final_df['geometry'].to_wkt()
    final_df = pd.DataFrame(final_df)
    final_df['geometry'] = wkt_geometries
    
    # Create the output directory if it doesn't exist.
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # Export the final, enriched DataFrame to a CSV file.
    final_df.to_csv(args.output_csv, index=False)
    
    # Print summary statistics to confirm the process was successful.
    print(f"Original input data shape: {orig_input_shape}")
    print(f"Original ADI data shape: {orig_adi_shape}")
    print(f"Successfully joined and saved {len(final_df)} records to {args.output_csv}. Final shape: {final_df.shape}")
