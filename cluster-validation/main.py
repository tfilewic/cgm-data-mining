#tfilewic  2025-09-24
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import sys

CGM_DATA_PATH = "CGMData.csv"
INSULIN_DATA_PATH = "InsulinData.csv"

INSULIN_FEATURES = ["Timestamp", "BWZ Carb Input (grams)"]
CGM_FEATURES = ["Timestamp", "Sensor Glucose (mg/dL)"]

def import_file(filename:str) -> pd.DataFrame:
    """
    reads csv into dataframe
    """
    print(f"Importing {filename} ...")
    try:
        return pd.read_csv(filepath_or_buffer=filename, header=0, low_memory=False)   #read data
    except Exception as e:
        sys.exit(f"ERROR: Failed to import {filename}: {e}")

def create_timestamps(df: pd.DataFrame) -> None:
    """
    derives Timestamp feature 
    from combines Date and Time columns 
    """
    date = df["Date"]
    time = df["Time"]
    timestamp = pd.to_datetime(date + " " + time, format='mixed')
    df.insert(0, "Timestamp", timestamp)

def select_features(features: list[str], df: pd.DataFrame) -> None:
    """
    removes features that are not listed in the features param
    """
    df.drop(columns=[column for column in df.columns if column not in features], inplace=True)


def get_meals(df: pd.DataFrame) -> pd.DataFrame:
    """
    keeps only start times of eligible meals
    """
    meals = df.copy()

    #drop NaNs
    meals.dropna(subset=["BWZ Carb Input (grams)"], inplace=True)

    #drop 0s
    meals.drop(meals[meals["BWZ Carb Input (grams)"] == 0].index, inplace=True)

    #drop meals which are followed by another meal within 2 hours
    too_soon = (meals["Timestamp"].shift(1) - meals["Timestamp"] <= pd.Timedelta("2h"))
    meals.drop(meals.index[too_soon], inplace=True)

    #drop carb input column
    # meals.drop(columns=["BWZ Carb Input (grams)"], inplace=True)

    #fix index
    meals.reset_index(drop=True, inplace=True)

    return meals

def get_nomeals(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates start times of eligible postabsorptive windows
    """
    meals = df.copy()

    #drop NaNs
    meals.dropna(subset=["BWZ Carb Input (grams)"], inplace=True)

    #drop 0s
    meals.drop(meals[meals["BWZ Carb Input (grams)"] == 0].index, inplace=True)

    nomeals = []
    for this_meal, next_meal in zip(meals["Timestamp"], meals["Timestamp"].shift(1)):
        
        #skip first row edge case
        if pd.isna(next_meal): 
            continue
            
        start = this_meal + pd.Timedelta("2h")
        end = next_meal - pd.Timedelta("2h")

        while (start < end):
            nomeals.append(start)
            start += pd.Timedelta("2h")

    return pd.DataFrame({"Timestamp": nomeals})
  

def build_meal_matrix(meals: pd.DataFrame, cgm: pd.DataFrame) -> np.array:
    """
    builds 30 sample absorptive windows from meal times 
    """
    THRESHOLD = 2
    matrix = []

    for meal in meals["Timestamp"]:
        window = cgm[(cgm["Timestamp"] >= meal - pd.Timedelta("30min")) &
                     (cgm["Timestamp"] <= meal + pd.Timedelta("2h"))]

        values = window["Sensor Glucose (mg/dL)"].to_numpy()

        #skip windows that dont have 30 pts
        if len(values) != 30:
            continue
        
        #discard if missing data points exceed threshold
        missing = np.isnan(values).sum()
        if missing > THRESHOLD:
            continue
        
        #fill missing
        if missing > 0:
            series = pd.Series(values)
            values = series.interpolate(limit_direction="both").to_numpy()

        #insert row
        matrix.append(values)

    return np.array(matrix, dtype=float)

def build_nomeal_matrix(nomeals: pd.DataFrame, cgm: pd.DataFrame) -> np.array:
    """
    builds 24 sample postabsorptive windows from nomeal times 
    """
    THRESHOLD = 2
    matrix = []

    for nomeal in nomeals["Timestamp"]:
        window = cgm[(cgm["Timestamp"] >= nomeal) &
                     (cgm["Timestamp"] <= nomeal + pd.Timedelta("2h"))]

        values = window["Sensor Glucose (mg/dL)"].to_numpy()

        #skip windows that dont have 24 pts
        if len(values) != 24:
            continue
        
        #discard if missing data points exceed threshold
        missing = np.isnan(values).sum()
        if missing > THRESHOLD:
            continue
        
        #fill missing
        if missing > 0:
            series = pd.Series(values)
            values = series.interpolate(limit_direction="both").to_numpy()

        #insert row
        matrix.append(values)

    return np.array(matrix, dtype=float)



''' PREPROCESSING '''
#import insulin data
cgm = import_file(CGM_DATA_PATH)
insulin = import_file(INSULIN_DATA_PATH)

#create timestamps
create_timestamps(cgm)
create_timestamps(insulin)

select_features(CGM_FEATURES, cgm)
select_features(INSULIN_FEATURES, insulin)

meals = get_meals(insulin)
nomeals = get_nomeals(insulin)

print(meals)

meal_matrix = build_meal_matrix(meals, cgm)
nomeal_matrix = build_nomeal_matrix(nomeals, cgm)
#p2 meal matrix   PxF




#print(meal_matrix)
# min/max carb 
# split into n bins - bin sz 20 grams, create bins (mincarb to min+20, min+21 to min+40, .... to maxcarb)
# Px1 bin matrix  is ground truth

#implement 2 clustering algos