import os
import pandas as pd
import timeit
import geopandas 
import boto3


def combine_surveys():
    """Loop over folders and files in current directory. 
    Combine files with .dta extension and return dataframe."""
    
    surveys = pd.DataFrame()
    
    s3 = boto3.client("s3")
    
    bucket = "w210-poverty-mapper"
    directory_path = "dhs_data/raw_data/"
    
    contents = s3.list_objects(Bucket=bucket, Prefix=directory_path)['Contents']

    directory_items = []

    for f in contents:
        directory_items.append(f["Key"])
        

    result_dta = [x for x in directory_items if ".DTA" in x]
    print(result_dta)
    
    for item in result_dta:
        print(item)
        df = pd.read_stata("s3://w210-poverty-mapper/" + item, convert_categoricals=False) # argument handles issue w/PH label
        surveys = surveys.append(df, ignore_index=True, sort=False)
    
    return surveys

        
        
def combine_gps():
    """Loop over folders and files in current directory. 
    Combine files with .shp extension and return geodataframe."""
    
    gps = geopandas.GeoDataFrame()

    s3 = boto3.client("s3")
    
    bucket = "w210-poverty-mapper"
    directory_path = "dhs_data/raw_data/"
    
    contents = s3.list_objects(Bucket=bucket, Prefix=directory_path)['Contents']

    directory_items = []

    for f in contents:
        directory_items.append(f["Key"])
        

    result_shp = [x for x in directory_items if x.endswith(".shp")]
    print(result_shp)
    
    for item in result_shp:
        gdf = geopandas.read_file("s3://w210-poverty-mapper/" + item)
        gps = gps.append(gdf, ignore_index=True, sort=False)
    
    return gps


def merge_surveys_gps(combined_surveys, combined_gps):
    """Align country code and cluster number data formats
    in combined_survey and combined_gps data and merge 
    datasets on country code and cluster number."""
    
    # Extract combined_surveys country code
    combined_surveys["dhscc"]=combined_surveys["hv000"].str.slice(stop=2)
    
    # Align merge column data types
    combined_surveys["dhscc"] = combined_surveys["dhscc"].astype("str")
    combined_surveys["hv001"] = combined_surveys["hv001"].astype("int")
    combined_gps["DHSCC"] = combined_gps["DHSCC"].astype("str")
    combined_gps["DHSCLUST"] = combined_gps["DHSCLUST"].astype("int")
    
    # Merge datasets on country code and cluster number 
    combined_dhs = combined_surveys.merge(combined_gps, how='left', 
                                      left_on=["dhscc", "hv001"], 
                                      right_on=["DHSCC", "DHSCLUST"])
    
    return combined_dhs