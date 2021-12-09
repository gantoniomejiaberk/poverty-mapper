#%pip install folium

import folium
import pandas as pd
import ujson


def tile_check_map(zone, lat, lon, start=None, end=None):
    """Plot zone tile bounding boxes from metadata."""
    
    meta_data = pd.read_csv("/root/tiles/" + zone + "/meta_data_" + zone + ".csv")
    
    # Check tile bounding boxes
    m = folium.Map([lat, lon], zoom_start=7)
    
    if start is None and end is None:

        for index, row in meta_data.iterrows():

        # Get tile bounding box
            tile_box = [(coords[1], coords[0]) for coords in ujson.loads(row["lat_lon_bounds"].replace("(", "[").replace(")", "]"))]
            folium.Rectangle(bounds=tile_box, color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(m)

        return m
    
    else:
        
        for index, row in meta_data[start:end].iterrows():

        # Get tile bounding box
            tile_box = [(coords[1], coords[0]) for coords in ujson.loads(row["lat_lon_bounds"].replace("(", "[").replace(")", "]"))]
            folium.Rectangle(bounds=tile_box, color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(m)

        return m