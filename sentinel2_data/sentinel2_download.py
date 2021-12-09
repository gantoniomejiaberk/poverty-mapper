#%pip install geopandas
#%pip install rasterio
#%pip install utm

import boto3
import sagemaker
from sagemaker import get_execution_role
import geopandas as gpd
from shapely.geometry.polygon import Polygon
import timeit
import shutil
import rasterio
import os


def to_zones(polygon): 
    """
    Take country polygon and return list of cooresponding UTM zones.
    """
    zone_min = int((polygon.bounds[0] + 180) / 6.0 + 1)
    zone_max = int((polygon.bounds[2] + 180) / 6.0 + 1)

    letters = ["C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]
    
    if polygon.bounds[1] > 80 or polygon.bounds[3] > 80: 
        raise Exception("Didn't correctly implement handling of X latitude band")

    lat_min = int((polygon.bounds[1] + 80) / 8.0)
    lat_max = int((polygon.bounds[3] + 80) / 8.0)

    zones = []
    for zone in range(zone_min, zone_max + 1): 
        for lat_ind in range(lat_min, lat_max + 1): 
            zones.append("{}{}".format(zone, letters[lat_ind]))
    
    return zones


def get_all_zones(country_list, country_shapes):
    """Get list of zones by country."""
    
    zone_set = set()

    for country in country_list:
        country_polygon = country_shapes.loc[
            country_shapes["CNTRY_NAME"] == country]["geometry"].iloc[0]
        country_zones = to_zones(country_polygon)
        for zone in country_zones:
            zone_set.add(zone)
    
    zone_list = sorted(list(zone_set))
    
    return zone_list


def get_sentinel_composite_data(zones): 
    """
    Get data for all countries in UTM dictionary.
    """
    for zone in zones:
        print(zone)
        os.system(f"""wget -r -nH --cut-dirs=6 --no-parent --reject="index.html*" http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/"$zone"/ -P /root""")
       

def move_to_s3(zones): 
    """"
    Write directories to s3 bucket
    """ 
    # Move directories to s3
    for zone in zones:
        start = timeit.default_timer()
        print("Writing {} to s3".format(zone))
        sess.upload_data("/root/" + zone, 
                         bucket=bucket, 
                         key_prefix=s3_path + "/" + zone)
    
        stop = timeit.default_timer()
        print("Time: ", round((stop - start)/60, 2))


def remove_from_sagemaker(zones):
    """
    Delete directories from Sagemaker "root" folder
    """
    for zone in zones:
        shutil.rmtree("/root/" + zone)
    

def list_s3_files(bucket, directory_path):
    """
    Print files in an s3 bucket directory.
    """
    contents = conn.list_objects(Bucket=bucket, 
                                 Prefix=directory_path)["Contents"]
    for f in contents:
        print(f["Key"])