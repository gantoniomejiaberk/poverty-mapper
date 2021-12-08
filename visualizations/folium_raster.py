import numpy as np

# Adapted from: https://stackoverflow.com/questions/14329691/convert-latitude-longitude-point-to-a-pixels-x-y-on-mercator-projection

def row_to_lat(row, a, b): 
    """Translate pixel indices to latitude using mercator projection."""
    lat = (360.0 / (2.0 * np.pi)) * 2.0 * (np.arctan(np.exp((row/a) - b)) - (np.pi / 4.0))
    return lat


def lat_to_row(lat, a, b): 
    """Inverse function to get latitude for a given row."""
    row = a * (np.log(np.tan((np.pi / 4.0) + ((2.0 * np.pi * (lat / 360.0)) / 2.0))) + b)
    return row