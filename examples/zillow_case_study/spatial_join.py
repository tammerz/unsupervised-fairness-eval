import pandas as pd
import geopandas as gpd
import argparse
import os

parser = argparse.ArgumentParser(description="Spatial Join Zillow Data with Census ADI")
parser.add_argument("--input_csv", type=str, default="Data/prepared_zillow_data.csv", help="Input path from data prep")
parser.add_argument("--output_csv", type=str, default="Data/la_county_prepared.csv", help="Output path for joined data")
args = parser.parse_args()

# Load your new CSV
df = pd.read_csv(args.input_csv)
orig_input_shape = df.shape

# Convert the text strings back into live spatial geometry
# Note: Coordinates were already scaled by 1,000,000 in prepare_zillow_input_data.py
df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])

# Re-initialize the GeoDataFrame
zillow_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# 3. Load the 2015 Census Block Group Shapefile
# Pass only the path to the .shp file. GeoPandas will automatically parse the .dbf, .shx, and .prj
shapefile_path = "Data/shapefile_2015_BlockGroup_Calif/tl_2015_06_bg.shp"
census_gdf = gpd.read_file(shapefile_path)

# 4. Standardize Coordinate Reference Systems (CRS)
# Shapefiles often use NAD83 (EPSG:4269). They must match perfectly to intersect.
census_gdf = census_gdf.to_crs(zillow_gdf.crs)

# 5. Execute the Spatial Join (Point-in-Polygon)
# This evaluates which block group boundary polygon encloses each property point
joined_gdf = gpd.sjoin(zillow_gdf, census_gdf, how="left", predicate="intersects")

# Drop duplicate matches in case a property point falls exactly on a boundary
joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]
joined_gdf = joined_gdf.drop(columns=['index_right'], errors='ignore')

# 6. Load and Merge the Area Deprivation Index (ADI) Dataset
# Force FIPS to remain string types to preserve leading zeros
adi_df = pd.read_csv('Data/adi_data_2015_calif_FIPS/CA_2015_ADI_Census Block Group_v3.1.csv', dtype={'FIPS': str})
orig_adi_shape = adi_df.shape

# Robust matching: Strip whitespace, handle 'nan', remove 'G' prefix, and pad to 12 digits
joined_gdf['GEOID'] = joined_gdf['GEOID'].astype(str).str.strip().str.replace('G', '', regex=False)
joined_gdf['GEOID'] = joined_gdf['GEOID'].apply(lambda x: x.zfill(12) if x.lower() != 'nan' else x)

adi_df['FIPS'] = adi_df['FIPS'].astype(str).str.strip().str.replace('G', '', regex=False).str.zfill(12)

# Combine the spatial results with the socioeconomic ranks
final_df = joined_gdf.merge(adi_df, left_on='GEOID', right_on='FIPS', how='left')

# 7. Clean the ADI data for the C4F tool pipeline
# Convert text suppression codes to NaN, drop them, and cast to integers
final_df['ADI_STATERNK'] = pd.to_numeric(final_df['ADI_STATERNK'], errors='coerce')
final_df = final_df.dropna(subset=['ADI_STATERNK'])
final_df['ADI_STATERNK'] = final_df['ADI_STATERNK'].astype(int)

if final_df.empty:
    print("ERROR: Resulting dataset is empty! No properties matched the ADI data.")
    print("Sample GEOIDs from spatial join:", joined_gdf['GEOID'].dropna().head().tolist())
    print("Sample FIPS from ADI data:", adi_df['FIPS'].head().tolist())
else:
    # Ensure it is explicitly typed as a GeoDataFrame so to_wkt() executes correctly
    if not isinstance(final_df, gpd.GeoDataFrame):
        final_df = gpd.GeoDataFrame(final_df, geometry='geometry')
        
    # Convert geometry back to WKT explicitly for safe CSV storage
    wkt_geometries = final_df['geometry'].to_wkt()
    final_df = pd.DataFrame(final_df)
    final_df['geometry'] = wkt_geometries
    
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # Export the ready-to-audit dataframe
    final_df.to_csv(args.output_csv, index=False)
    print(f"Original input data shape: {orig_input_shape}")
    print(f"Original ADI data shape: {orig_adi_shape}")
    print(f"Successfully joined and saved {len(final_df)} records to {args.output_csv}. Final shape: {final_df.shape}")