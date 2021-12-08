import pandas as pd
import numpy as np
import timeit


def get_counts(col_list): 
    """Take a list of columns and print value counts, including NAN, 
    for each column in the list."""
    for col in col_list:
        print(combined_dhs_subset[col].value_counts(dropna=False))
        print("-"*20)

    
def cat_recode(value, codes):
    """Categorical recode logic to use in "recode" function below."""
    if pd.isna(value): 
        return np.nan
    elif value in codes:
        return 1
    else:
        return 0

    
def range_recode(value, lower, upper):
    """Range recode logic to use in "recode" function below."""
    if pd.isna(value): 
        return np.nan
    elif lower <= value <= upper:
        return 1
    else:
        return 0


def multi_recode(row):
    """Multiple variable recode logic to use in "recode" function below."""
    if row.isna().astype(int).prod() == 1: 
        return np.nan
    elif row.sum() >= 1:
        return 1
    elif row.isna().any():
        return np.nan
    else: 
        return 0
    
    
def recode(dict_list):
    """Take a list of recode dictionaries and perform recodes 
    according to dictionary inputs. Category dictionaries are 
    recoded to one indicator per category group. Range dictionaries 
    are recoded to one indicator per range group. Multi-variable
    dictionaries are recoded to one indicator based on the values 
    of the variables in the "var" list."""
    
    for item in dict_list:
        if item["dict_type"] == "cat":
            for name in item["types"]:
                combined_dhs_subset[name] = combined_dhs_subset[item["var"]].apply(
                    lambda value: cat_recode(value, item["types"][name]))
        if item["dict_type"] == "range":
            for name in item["types"]:
                combined_dhs_subset[name] = combined_dhs_subset[item["var"]].apply(
                    lambda value: range_recode(value, 
                                               item["types"][name][0], 
                                               item["types"][name][1]))
        elif item["dict_type"] == "multi_var":
            for name in item["types"]:
                combined_dhs_subset[name] = combined_dhs_subset[item["var"]].apply(
                    multi_recode, axis=1)
                
                
def spot_check(dict_name, num_rows):
    """Enter name of an indicator dictionary and a number of rows
    and print first and last number of rows for source column(s) 
    and derived indicator column(s)."""
    if (dict_name["dict_type"] == "cat") or (dict_name["dict_type"] == "range"):
        cols = [key for key in dict_name["types"].keys()]
        cols.append(dict_name["var"])
        print(combined_dhs_subset[cols].head(num_rows))
        print(combined_dhs_subset[cols].tail(num_rows))
    elif dict_name["dict_type"] == "multi_cat":
        cols = dict_name["var"]
        cols.extend([key for key in dict_name["types"].keys()])
        print(combined_dhs_subset[cols].head(num_rows))
        print(combined_dhs_subset[cols].tail(num_rows))
        
           
def check_cat_recodes(col_list): 
    """Take a list of columns that were recoded to categories 
    and check the derived indicators for double or skipped counting."""
    for item in col_list:
        combined_dhs_subset["col_sum"] = combined_dhs_subset[
            [item + "_low", item + "_med", item + "_high"]].sum(axis=1)
        print("{} not counted: {}".format(
            item, combined_dhs_subset[(combined_dhs_subset["col_sum"] == 0) & 
                                             (combined_dhs_subset[item].notna())].shape[0]))
        print("{} double counted: {}".format(
            item, combined_dhs_subset[combined_dhs_subset["col_sum"] > 1].shape[0]))