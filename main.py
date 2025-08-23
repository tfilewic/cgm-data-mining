import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

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
        sys.exit(f"ERROR: Failed to import {filename}: {e}")

    
def select_features(features: list[str], df: pd.DataFrame) -> None:
    """
    removes feature that are not listed in the features param in place 
    """
    df.drop(columns=[column for column in df.columns if column not in features], inplace=True)

def interpolate_glucose(df: pd.DataFrame) -> None:
    """
    linearly interpolates missing cgm values in place
    """
    df["Sensor Glucose (mg/dL)"].interpolate(method="linear", inplace=True)

def create_timestamps(df: pd.DataFrame) -> None:
    """
    combines date and time columns into single Timestamp feature and drops the original columns (in place)
    """
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df.drop(columns=["Date", "Time"], inplace=True)

def create_period(df: pd.DataFrame) -> None:
    """
    discretizes time into Period feature for Daytime (06:00:00 to 23:59:59) or Overnight (00:00:00 to 05:59:59)
    """
    hours = df["Timestamp"].dt.hour
    df["Period"] = np.where(hours >= 6, "Daytime", "Overnight")
    df.insert(0, "Period", df.pop("Period"))

def create_mode(df: pd.DataFrame) -> None:
    """
    creates Mode feature for Manual or Auto based on first appearance of "AUTO MODE ACTIVE PLGM OFF" in Insulin df
    """
    AUTO_ON = "AUTO MODE ACTIVE PLGM OFF"
    auto_start_timestamp = insulin.loc[insulin["Alarm"] == AUTO_ON, "Timestamp"].min()  #timestamp of earliest auto on alarm
    df["Mode"] = np.where(df["Timestamp"] >= auto_start_timestamp, "Auto", "Manual")

def create_day(df: pd.DataFrame) -> None:
    """
    replaces Timestamp with Day daytime.date object
    """
    df["Day"] = df["Timestamp"].dt.date
    df.drop(columns=["Timestamp"], inplace=True)
    df.insert(0, "Day", df.pop("Day"))

''' Import data '''
#parse files to dfs
cgm = import_file(CGM_DATA_PATH)
insulin = import_file(INSULIN_DATA_PATH)

      
''' Preprocesing '''
#feature subset selection
select_features(CGM_SUBSET, cgm)
select_features(INSULIN_SUBSET, insulin)

#interpolate missing values
interpolate_glucose(cgm)

#create new Timestamp feature from Date and Time
create_timestamps(cgm)
create_timestamps(insulin)

#create Mode feature (Manual or Auto)
create_mode(cgm)

#discretize time into Period (Daytime or Overnight)
create_period(cgm)

#create Day feature from Timestamp
create_day(cgm)


''' Analysis ''' 
#TODO



''' Export data '''
#TODO