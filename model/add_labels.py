#!pip install tdqm geopy

import pandas as pd
import numpy as np
import boto3
from tqdm.notebook import trange, tqdm
from math import cos, radians, nan
import geopy.distance


def km_to_lat_lon_displacement(km, origin_latitude):
    lat = km/111.111
    lon = km/(111.111 * cos(radians(origin_latitude)))
    
    return lat, lon


def calculate_weighted_index(center, data, radius):
    """Take a center coordinate, dataset with wealth indices, 
    and radius parameter and calculate the distance-based 
    weighted average for a satellite image tile."""
    try:
        lon, lat = tuple(map(float, center.strip("()").split(', ')))
    except:
        print("ERROR: unable to extract latitude and logitude data from the center column")
    
    weighted_index = nan
    point_count = 0
    index_range = nan
    
    latOffset, lonOffset = km_to_lat_lon_displacement(radius + 1, lat)
    
    first_subset = data[
        ((lat - latOffset)  < data['lat']) & (data['lat'] < (lat + latOffset)) &
        ((lon - lonOffset)  < data['lon']) & (data['lon'] < (lon + lonOffset))
    ]
    
    if len(first_subset) > 0:
        first_subset["distance"] = first_subset.apply(lambda inner_row: geopy.distance.distance((lat, lon), (inner_row["lat"], inner_row["lon"])).km, axis=1)
        inside_radius = first_subset[first_subset["distance"] < radius]
        
        if len(inside_radius) > 0:
            ##### Weighted average calculations are done here
            min_wealth_index = np.min(inside_radius.wealth_index)
            max_wealth_index = np.max(inside_radius.wealth_index)
            min_distance = np.min(inside_radius.distance)
            max_distance = np.max(inside_radius.distance)
            index_range =  max_wealth_index - min_wealth_index
            point_count = len(inside_radius)

            if min_distance == 0.0: #case where its close or the same coordinate (zero)
                inside_radius.loc[inside_radius["distance"] < 0.01,'distance'] = 0.01

            inverse_weight = radius / inside_radius.distance
            inside_radius["weight"] = inverse_weight

            #This is the weighted calculation
            weighted_index = np.sum((inside_radius.wealth_index * inside_radius.weight)) / np.sum(inside_radius.weight)

            # to remove confusion, weighted_index is set to NaN when there are no points
            if point_count == 0:
                weighted_index = nan
        
        
    return weighted_index, point_count, index_range


def add_labels(radius, meta_data, survey_data):
    """
    Take a radius, meta dataset, and survey data with wealth indices.
    Calcuate distance weighted average wealth index using survey data 
    within radius. Return meta dataset with weighted average index.
    """
    
    labeled_data = meta_data.copy()
    tqdm.pandas(desc="Weighted Calculation for Radius " + str(radius))
    
    # Add weighted average wealth index
    labeled_data[["weighted_index", "point_count", "index_range"]] = labeled_data.progress_apply(lambda row: calculate_weighted_index(row["center"], survey_data, radius), axis=1,result_type='expand')
    
    # Remove s3 path
    labeled_data['filename'] = labeled_data['filename'].str.replace("sentinel2_composite/transformed_data/", "")
    
    return labeled_data


def add_bins(labeled_data, bin_type, num_classes, quantile_list=False):
    """
    Take a labeled dataset, keep records with a weighted wealth 
    index and bin wealth indexes using method specified by 
    bin_type and num_classes. Return dataset with classes.
    
    num_classes:      integer for number of classes (must be greater than one)
    bin_type:         string indicating whether to bin ("within", "across") countries
    quantile_list:    list specifying quantile boundaries for two class case, e.g. [0, .20, 1]
    """
    
    # Keep records with wealth index
    labels_only = labeled_data[labeled_data.point_count > 0]
    
    if num_classes > 2:

        if bin_type == "across":
            labels_only["label_name"] = pd.qcut(labels_only["weighted_index"], q=num_classes)
            labels_only ["label"] = pd.qcut(labels_only["weighted_index"], q=num_classes, labels=False)

        if bin_type == "within":
            labels_only["label_name"] = labels_only.groupby(
                ["countries"])["weighted_index"].apply(lambda x: pd.qcut(x, q=num_classes))
            labels_only["label"] = labels_only.groupby(
                ["countries"])["weighted_index"].apply(lambda x: pd.qcut(x, q=num_classes, 
                                                                             labels=False))
    if num_classes == 2:

        if bin_type == "across":
            labels_only["label_name"] = pd.qcut(labels_only["weighted_index"], q=quantile_list)
            labels_only ["label"] = pd.qcut(labels_only["weighted_index"], q=quantile_list, labels=False)

        if bin_type == "within":
            labels_only["label_name"] = labels_only.groupby(
                ["countries"])["weighted_index"].apply(lambda x: pd.qcut(x, q=quantile_list))
            labels_only["label"] = labels_only.groupby(
                ["countries"])["weighted_index"].apply(lambda x: pd.qcut(x, q=quantile_list, 
                                                                             labels=False))
    return labels_only