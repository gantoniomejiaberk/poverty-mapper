#%pip install geopandas
#%pip install rasterio
#%pip install utm

##%%capture
#!apt update
#!apt install -y software-properties-common
#!apt install -y gpg
#!add-apt-repository -y ppa:ubuntugis/ppa && apt-get update
#!add-apt-repository -y ppa:nextgis/ppa && apt-get update 
#!apt-get install -y gdal-bin
#!apt-get install -y libgdal-dev pgp
#!export CPLUS_INCLUDE_PATH=/usr/include/gdal
#!export C_INCLUDE_PATH=/usr/include/gdal
#!apt-get install -y --reinstall build-essential
#!pip3 install "setuptools<58"
#!pip3 install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`


import boto3
import sagemaker
from sagemaker import get_execution_role
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.prepared import prep
import timeit
import shutil
from tqdm import tqdm
import rasterio
from rasterio.plot import show_hist
from rasterio.plot import show
from rasterio.windows import Window
import utm
import pyproj
import os
import pandas as pd
import geopandas as gpd
import random
from osgeo import gdal
import time


# Tile generation adapted from: https://fractaldle.medium.com/satellite-images-deep-learning-spacenet-building-segmentation-a5d145a81c33


class UTM2LonLat(object): 
    def __init__(self, rasterio_image, verbose=False):
        self.transform = rasterio_image.transform
        self.epsg = int(str(rasterio_image.crs)[-5:])
        self.verbose = verbose
        
        #self.coordinate_proj = pyproj.Proj(proj='utm', zone=self.zone, ellps='WGS84')
        self.coordinate_proj = pyproj.Proj("epsg:{}".format(self.epsg))
        
    def index_to_lon_lat (self, index_x, index_y):
        x, y = self.transform * (index_x, index_y)
        if self.verbose:
            print("UTM1: {} -> {}".format((index_x, index_y), (x, y)))
        lon, lat = self.coordinate_proj(x, y, inverse=True)
        if self.verbose: 
            print("UTM2: {} -> {}".format((x, y), (lon, lat)))
        return (lon, lat) 
    
    def get_epsg(self):
        return self.epsg
    
    def get_transform(self): 
        return self.transform
   

def get_lon_lat_bounds(conv, pixel_positions=None):
    """
    Get lon/lat bounding box coordinates
    """
    
    if pixel_positions is None: 
        top_left = conv.index_to_lon_lat(0, 0)
        bottom_left = conv.index_to_lon_lat(0, tif.height)
        top_right = conv.index_to_lon_lat(tif.width, 0)
        bottom_right = conv.index_to_lon_lat(tif.width, tif.height)
    
        return [top_left, top_right, bottom_right, bottom_left]
    else: 
        return [conv.index_to_lon_lat(pos[0], pos[1]) for pos in pixel_positions]
    

def get_filename(x_index, y_index, zone, geo):
    """
    Generate index specific tile name
    """
    filename = "{}_{}_{}_{}.tif".format(zone, geo, str(x_index).zfill(5), str(y_index).zfill(5))
    
    return filename


def get_tile_transform(parent_transform, pixel_x, pixel_y):
    """
    Create tile transform matrix from parent tif image
    """
    crs_x = parent_transform.c + pixel_x * parent_transform.a
    crs_y = parent_transform.f + pixel_y * parent_transform.e
    tile_transform = rasterio.Affine(parent_transform.a, parent_transform.b, crs_x,
                                     parent_transform.d, parent_transform.e, crs_y)
    return tile_transform
 
    
def get_tile_profile(parent_tif, pixel_x, pixel_y):
    """
    Prepare tile profile
    """
    tile_crs = parent_tif.crs
    tile_nodata = parent_tif.nodata if parent_tif.nodata is not None else 0
    tile_transform = get_tile_transform(parent_tif.transform, pixel_x, pixel_y)
    profile = dict(
                driver="GTiff",
                crs=tile_crs,
                nodata=tile_nodata,
                transform=tile_transform
            )
    
    return profile


def get_countries(polygon_dict, corners): 
    """
    Check whether tile corner coordinates are within any of the
    polygons in polygon_dict.
    """
    countries = []
    
    points = []
    for coordinate in corners: 
        points.append(Point(coordinate[0], coordinate[1]))

    for country in polygon_dict:
        #t = time.time()
        #hits = list(filter(polygon_dict[country].contains, points))
        #print(hits)
        #if len(hits) > 0:
        #    countries.append(country)
        for point in points:
            if polygon_dict[country].contains(point):
                countries.append(country)
                break
        #times[country] = {"time": time.time() - t, "corners": corners}
                
    return countries


def get_utm_zone_polygon(zone): 
    letters = ["C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", 
               "P", "Q", "R", "S", "T", "U", "V", "W", "X"]
    
    number = int(zone[:-1])
    letter = zone[-1]
    letter_number = letters.index(letter) + 1
    
    east =  (number * 6) - 180
    west = east - 6
    north = (letter_number * 8) - 80
    south = north - 8
    
    utm_zone_polygon = Polygon([[west, north], [east, north], [east, south], [west, south]])
 
    return utm_zone_polygon


def all_in_utm_zone(utm_zone_polygon, corners):
    """
    Take a UTM polygon and a list of tile corner coordintes and check
    whether all corner coordinates are within the UTM zone.
    """

    for coordinate in corners:
        point = Point(coordinate[0], coordinate[1])
        if not utm_zone_polygon.contains(point): 
            return False

    # Only return true if all corners didn't return false before
    return True


def generate_tiles(conn, bucket, zone_list, size, polygon_dict, s3_path):
    """
    Cut a .tif file into 224x224 tiles, write the tiles
    as .tif and .png and capture meta data.
    """
    
    # Convert .tif to .png
    png_options_list = ['-ot Byte',
                    '-of PNG',
                    '-b 1 -b 2 -b 3',
                    '-scale'
                   ]           

    png_options_string = " ".join(png_options_list)
    
    for zone in zone_list:
        meta_data = []

        zone_polygon = prep(get_utm_zone_polygon(zone))
        
        contents = conn.list_objects(
            Bucket=bucket, 
            Prefix="sentinel2_composite/raw_data/" + zone)["Contents"]
        
        for file in contents:
            if file["Key"].endswith(".tif"):
                print("Processing {}...".format(file["Key"]))

                geo = file["Key"].split("/")[3].split("_")[-1].split(".")[0]
                zone_number = file["Key"].split("/")[2][0:2]
                tif = rasterio.open("s3://w210-poverty-mapper/" + file["Key"])
                conv = UTM2LonLat(tif)

                i = 0

                for x in range(0, tif.width, size):
                    i += 1
                    j = 0

                    print(i)
                    for y in range(0, tif.height, size):
                        j += 1
                        
                        #t = time.time()
                        
                        pixel_positions = [[x, y], [x, y+size], [x+size, y], [x+size, y+size]]
                        corners = get_lon_lat_bounds(conv, pixel_positions)
                        
                        #elapsed = time.time() - t
                        #print("Corners:   {:.4f}".format(elapsed))
                        #t = time.time()
                        
                        countries = get_countries(polygon_dict, corners)
                        
                        #elapsed = time.time() - t
                        #print("Countries: {:.4f}".format(elapsed))
                        #t = time.time()

                        # Skip if this tile is not contained in any country
                        if len(countries) == 0: 
                            continue
                        
                        # Skip if not fully within UTM zone
                        if not all_in_utm_zone(zone_polygon, corners):
                            continue
                        
                        #elapsed = time.time() - t
                        #print("UTM check: {:.4f}".format(elapsed))
                        #t = time.time()
                        
                        # Creating tile specific profile
                        profile = get_tile_profile(tif, x, y)
                        
                        # Extract pixel data
                        tile_data = tif.read(window=((y, y + size), (x, x + size)),
                                             boundless=True, fill_value=profile['nodata'])[:3]
                        
                        partial = False
                        if (y + size > tif.height) or (x + size > tif.width): 
                            partial = True                   
                        
                        c, h, w = tile_data.shape
                        profile.update(
                            height=h,
                            width=w,
                            count=c,
                            dtype=tile_data.dtype,
                        )
                        
                        # Get filename and file paths
                        filename = get_filename(i, j, zone, geo)
                        root_path = "/root/tiles/" + zone + "/" + geo 
                        
                        if not os.path.exists(root_path): 
                            os.makedirs(root_path)
                        
                        root_tile_path = root_path + "/" + filename
                        s3_tile_path = s3_path + "/" + zone + "/" + geo + "/" + filename
                        
                        #elapsed = time.time() - t
                        #print("Cut tif:   {:.2f}".format(elapsed))
                        #t = time.time()
                        
                        # Write .tif and capture metadata
                        with rasterio.open(root_tile_path, "w", **profile) as dst:
                            dst.write(tile_data)

                            meta_data.append({"filename": s3_tile_path.split(".")[0], 
                                              "zone": zone, 
                                              "center": dst.lnglat(), 
                                              "lat_lon_bounds": corners,
                                              "utm_bounds": dst.bounds, 
                                              "partial": partial, 
                                              "countries": countries})
                        
                        #elapsed = time.time() - t
                        #print("Save tif:  {:.2f}".format(elapsed))
                        #t = time.time()
                        
                        png_tile_path = root_tile_path.split(".")[0] + ".png"
    
                        gdal.Translate(
                            png_tile_path,
                            root_tile_path,
                            options=png_options_string
                        ) 
        
                        #elapsed = time.time() - t
                        #print("Saved png: {:.2f}".format(elapsed))
                        #print("")
                        #t = time.time()
            
        # Only write if we created the directory above
        if os.path.exists("/root/tiles/" + zone): 
            # Write metadata to csv 
            df = pd.DataFrame(data=meta_data)

            csv_path = "/root/tiles/{}/meta_data_{}.csv".format(zone, zone)
            df.to_csv(csv_path, index=False)

            
def count_files_in_subd(zone):
    """
    Check root directory file count
    """
    for root, dirs, files in os.walk("/root/tiles/" + zone):
        print("{} in {}".format(len(files), root))

        
def move_tiles_to_s3(zone_list, s3_save_path): 
    """"
    Write directories to s3 bucket
    """ 
    # Move directories to s3
    for zone in tqdm(zone_list):
        start = timeit.default_timer()
        print("Writing {} to s3".format(zone))
        sess.upload_data("/root/tiles/" + zone, 
                         bucket=bucket, 
                         key_prefix=s3_save_path + "/" + zone)
    
        stop = timeit.default_timer()
        print("Time: ", round((stop - start)/60, 2))