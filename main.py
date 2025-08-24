#tfilewic  2025-08-24

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

CGM_DATA_PATH = "CGMData.csv"
INSULIN_DATA_PATH = "InsulinData.csv"
RESULTS_PATH = "Result.csv"

GLUCOSE_CATEGORIES = ["Hyper", "Hyper Critical", "In Range Primary", "In Range Secondary", "Hypo 1", "Hypo 2"]
CGM_SUBSET = ["Mode", "Day", "Period"] + GLUCOSE_CATEGORIES
INSULIN_SUBSET = ["Timestamp", "Alarm"]


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

def discretize_glucose(df: pd.DataFrame) -> None:
    """
    derives boolean glucose categories from Sensor Glucose (mg/dL)
    """
    glucose = df["Sensor Glucose (mg/dL)"]

    df["Hyper"] = glucose > 180
    df["Hyper Critical"] = glucose > 250
    df["In Range Primary"] = (70 <= glucose) &  (glucose <= 180)
    df["In Range Secondary"] = (70 <= glucose) &  (glucose <= 150)
    df["Hypo 1"] = glucose < 70
    df["Hypo 2"] = glucose < 54

def create_timestamps(df: pd.DataFrame) -> None:
    """
    derives Timestamp feature from combines Date and Time columns 
    """
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])

def create_period(df: pd.DataFrame) -> None:
    """
    derives Period feature (Daytime (06:00:00 to 23:59:59) or Overnight (00:00:00 to 05:59:59)) from Timestamp
    """
    hours = df["Timestamp"].dt.hour
    df["Period"] = np.where(hours >= 6, "Daytime", "Overnight")
    df.insert(0, "Period", df.pop("Period"))

def create_mode(df: pd.DataFrame) -> None:
    """
    derives Mode feature (Manual or Auto) based on first appearance of "AUTO MODE ACTIVE PLGM OFF" in Insulin df
    """
    AUTO_ON = "AUTO MODE ACTIVE PLGM OFF"
    auto_start_timestamp = insulin.loc[insulin["Alarm"] == AUTO_ON, "Timestamp"].min()  #timestamp of earliest auto on alarm
    df["Mode"] = np.where(df["Timestamp"] >= auto_start_timestamp, "Auto", "Manual")
    df.drop(df[df["Timestamp"].dt.date == auto_start_timestamp.date()].index, inplace=True)  #handle partial day dilution by dropping switch date

def create_day(df: pd.DataFrame) -> None:
    """
    derives Day (daytime.date object) from Timestamp
    """
    df["Day"] = df["Timestamp"].dt.date
    df.insert(0, "Day", df.pop("Day"))

def create_row(mode: str) -> list[float]:
    """
    generates a row for export
    """
    overnight = mean_period.loc[(mode, "Overnight"), GLUCOSE_CATEGORIES].tolist()
    daytime = mean_period.loc[(mode, "Daytime"), GLUCOSE_CATEGORIES].tolist()
    whole = mean_day.loc[mode, GLUCOSE_CATEGORIES].tolist()
    
    return overnight + daytime + whole

def export(rows: list[float], filename: str) -> None:
    """
    writes the results to csv
    """
    try:
        pd.DataFrame(rows).to_csv(filename, header=False, index=False)
    except Exception as e:
        sys.exit(f"ERROR: Failed to write {filename}: {e}")


''' Import data '''
#parse files to dfs
cgm = import_file(CGM_DATA_PATH)
insulin = import_file(INSULIN_DATA_PATH)

      
''' Preprocessing '''
#create new Timestamp feature from Date and Time
create_timestamps(cgm)
create_timestamps(insulin)

#interpolate missing values
interpolate_glucose(cgm)

#discretize glucose readings into boolean categories
discretize_glucose(cgm)

#feature subset selection for insulin
select_features(INSULIN_SUBSET, insulin)

#create Mode feature (Manual or Auto)
create_mode(cgm)

#discretize time into Period (Daytime or Overnight)
create_period(cgm)

#create Day feature from Timestamp
create_day(cgm)

#feature subset selection for cgm
select_features(CGM_SUBSET, cgm)


''' Analysis '''  
#analyze by day
mean_day = (cgm.groupby(["Mode", "Day"])[GLUCOSE_CATEGORIES].sum() / 288 * 100).groupby(level="Mode").mean()

#analyze by day-period
mean_period = (cgm.groupby(["Mode", "Day", "Period"])[GLUCOSE_CATEGORIES].sum() / 288 * 100).groupby(level=["Mode", "Period"]).mean()


''' Export data '''
manual = create_row("Manual")
auto = create_row("Auto")
export([manual, auto], RESULTS_PATH)