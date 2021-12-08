import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils import resample
import os
import shutil
import timeit


def get_even_samples(train):
    """
    Take a train dataframe and resample with replacement to get
    even classes. Return train dataframe with even classes.
    """
    value_counts = train["label"].value_counts()
    max_value = np.max(value_counts)
    for label, class_to_resample in value_counts[1:].items():
        n_samples = max_value - class_to_resample
        if n_samples > 0:
            data_for_label = train[train["label"] == label]
            additional_samples = resample(data_for_label, 
                                          replace=True,
                                          n_samples=n_samples,
                                          random_state=111)
            train = pd.concat([train, additional_samples])
    
    return train



def leave_one_out_csvs(meta_data, meta_data_name, directory, even=False):
    """
    Get train, validation, test sets for leave one out models. 
    Optionally, resample train set to even classes.
    """
    
    start = timeit.default_timer()
    
    if even == True:
        directory = directory + "_even"
    
    output_path = "s3://w210-poverty-mapper/modeling/data_splits/" + directory
    
    # Parse and lower country names
    meta_data["countries_parsed"] = meta_data["countries"].map(lambda x: x.lstrip("['").rstrip("'']").replace(" ", "_").lower())
    
    # Get unique list of countries
    country_list = [c for c in meta_data["countries_parsed"].unique()]
    
    for country in (country_list):
        
        # Get test/train images
        rest = meta_data[meta_data["countries_parsed"] == country]
        train = meta_data[meta_data["countries_parsed"] != country]
        
        val, test = train_test_split(rest, test_size=0.5)
        
        # Resample to get even classes
        if even == True:
            train = get_even_samples(train)

        # Shuffle train data
        train = shuffle(train)

        image_set = {"test": test,
                     "train": train, 
                     "val": val}
    
        for key, value in image_set.items():
            
            # Keep relevant columns
            subset = value[["filename", "label"]]
            
            csv_filename = "leave_one_out_" + country + "_" + meta_data_name + "_" + key + ".csv"
            
            # Write metadata s3  
            subset.to_csv(output_path + "/" + "leave_one_out_" + country + "_" + meta_data_name + "/" + csv_filename, index=False)
            print("Wrote {}".format(csv_filename))
                
    stop = timeit.default_timer()
    print('Time: ', round((stop - start)/60, 2))

 

def within_country_csvs(meta_data, meta_data_name, directory, even=False):
    """
    Get train, validation, test sets for within country models. 
    Optionally, resample train set to even classes.    
    """
    
    start = timeit.default_timer()
    
    if even == True:
        directory = directory + "_even"
    
    output_path = "s3://w210-poverty-mapper/modeling/data_splits/" + directory
    
    # Parse and lower country names
    meta_data["countries_parsed"] = meta_data["countries"].map(lambda x: x.lstrip("['").rstrip("'']").replace(" ", "_").lower())
    
    # Get unique list of countries
    country_list = [c for c in meta_data["countries_parsed"].unique()]
    
    for country in (country_list):
        print(country)
    
        # Select only images from country
        country_only = meta_data[meta_data["countries_parsed"] == country]

        # Get shuffled 70/15/15 train/val/test split
        train, rest = train_test_split(country_only, test_size=0.3)
        val, test = train_test_split(rest, test_size=0.5)
        
        # Resample to get even classes
        if even == True:
            train = get_even_samples(train)
            
            # Shuffle train data
            train = shuffle(train)

        image_set = {"test": test,
                     "train": train,
                     "val": val}

        for key, value in image_set.items():

            # Keep relevant columns
            subset = value[["filename", "label"]]

            csv_filename = "within_country" + "_" + country + "_" + meta_data_name + "_" + key + ".csv"

            # Write metadata s3  
            subset.to_csv(output_path + "/" + "within_country" + "_" + country + "_" + meta_data_name + "/" + csv_filename, index=False)
            print("Wrote {}".format(csv_filename))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start)/60, 2))
    
    
    
def leave_one_out_subset(meta_data, meta_data_name, split_name, hold_out_list, train_list, directory, even=False):
    """
    Get train, validation, test sets for leave-one-out 
    similar geography experiments. Train set includes 
    countries in "train_list". Validation and test set
    from country in hold_out_list. Optionally, resample 
    train set to even classes.
    """
    
    start = timeit.default_timer()
    
    if even == True:
        directory = directory + "_even"
    
    # Set output path
    output_path = "s3://w210-poverty-mapper/modeling/data_splits/" + directory 
    
    # Parse and lower country names
    meta_data["countries_parsed"] = meta_data["countries"].map(lambda x: x.lstrip("['").rstrip("'']").replace(" ", "_").lower())
         
    # Get test/train images
    rest = meta_data[meta_data["countries_parsed"].isin(hold_out_list)]
    train = meta_data[meta_data["countries_parsed"].isin(train_list)]
    
    val, test = train_test_split(rest, test_size=0.5)
    
    # Resample to get even classes
    if even == True:
        train = get_even_samples(train)

    # Shuffle train data
    train = shuffle(train)

    image_set = {"test": test,
                 "train": train, 
                 "val": val}
    
    for key, value in image_set.items():
        
        # Keep relevant columns
        subset = value[["filename", "label"]]
            
        csv_filename = split_name + "_" + meta_data_name + "_" + key + ".csv"
            
        # Write metadata s3  
        subset.to_csv(output_path + "/" + split_name + "_" + meta_data_name + "/" + csv_filename, index=False)
        print("Wrote {}".format(csv_filename))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start)/60, 2))
    

    
def cross_country_subset(meta_data, meta_data_name, split_name, subset_list, directory, even=False):
    """
    Get train, validation, test sets for cross-country experiments.
    Included countries are defined in "subset_list". Optionally, 
    resample train set to even classes.
    """
    
    start = timeit.default_timer()
    
    if even == True:
        directory = directory + "_even"
    
    # Set output path
    output_path = "s3://w210-poverty-mapper/modeling/data_splits/" + directory
    
    # Parse and lower country names
    meta_data["countries_parsed"] = meta_data["countries"].map(lambda x: x.lstrip("['").rstrip("'']").replace(" ", "_").lower())
         
    # Get test/train images
    subset = meta_data[meta_data["countries_parsed"].isin(subset_list)]
    
    # Get shuffled 70/15/15 train/val/test split
    train, rest = train_test_split(subset, test_size=0.3)
    val, test = train_test_split(rest, test_size=0.5)
    
    # Resample to get even classes
    if even == True:
        train = get_even_samples(train)

        # Shuffle train data
        train = shuffle(train)
        
    image_set = {"test": test,
                 "train": train,
                 "val": val}
    
    for key, value in image_set.items():
            
        # Keep relevant columns
        subset = value[["filename", "label"]]

        csv_filename = split_name + "_" + meta_data_name + "_" + key + ".csv"

        # Write metadata s3  
        subset.to_csv(output_path + "/" + split_name + "_" + meta_data_name + "/" + csv_filename, index=False)
        print("Wrote {}".format(csv_filename))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start)/60, 2))
    