import pandas as pd
import numpy as np
from pathlib import Path

CGM_DATA_PATH = "CGMData.csv"
INSULIN_DATA_PATH = "InsulinData.csv"

CGM_SUBSET = ["Date", "Time", "Sensor Glucose (mg/dL)"]
INSULIN_SUBSET = ["Date", "Time", "Alarm"]




def import_file(filename:str) -> pd.DataFrame:
    """
    reads csv into dataframe
    """
    print(f"Importing {filename} ...")
    try:
        return pd.read_csv(filepath_or_buffer=filename, header=0, low_memory=False)   #read data
    except Exception as e:
        print(f"ERROR: Failed to import {filename}: {e}")
        return pd.DataFrame()
    
def select_features(features: list[str], df: pd.DataFrame) -> None:
    """
    removes columns in place that are not listed in the features param
    """
    df.drop(columns=[column for column in df.columns if column not in features], inplace=True)



#parse files to dfs
cgm = import_file(CGM_DATA_PATH)
insulin = import_file(INSULIN_DATA_PATH)

#feature subset selection
select_features(CGM_SUBSET, cgm)
select_features(INSULIN_SUBSET, insulin)

#interpolate missing values
cgm["Sensor Glucose (mg/dL)"].interpolate(method="linear", inplace=True)