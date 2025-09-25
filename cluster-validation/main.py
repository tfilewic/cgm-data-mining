#tfilewic  2025-09-25
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

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

def build_meal_matrix(meals: pd.DataFrame, cgm: pd.DataFrame) -> tuple[np.array, np.array]:
    """
    builds 30 sample absorptive windows from meal times 
    """
    THRESHOLD = 2
    matrix = []
    labels = []

    for _, row in meals.iterrows():
        meal = row["Timestamp"]
        carb_val = row["BWZ Carb Input (grams)"]

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
        labels.append(carb_val)

    return np.array(matrix, dtype=float), np.array(labels, dtype=float)



''' IMPORT FILES '''
#import insulin data
cgm = import_file(CGM_DATA_PATH)
insulin = import_file(INSULIN_DATA_PATH)


''' BUILD MATRIX '''
create_timestamps(cgm)
create_timestamps(insulin)

select_features(CGM_FEATURES, cgm)
select_features(INSULIN_FEATURES, insulin)

meals = get_meals(insulin)

meal_matrix, carb_labels = build_meal_matrix(meals, cgm)
 
''' BUILD BINS '''
minimum = carb_labels.min()
maximum = carb_labels.max()
bins = np.floor((carb_labels - minimum) / 20).astype(int)
bin_count = bins.max() + 1

''' SCALE FEATURES '''
scaler = StandardScaler()
X = scaler.fit_transform(meal_matrix)


''' CLUSTERING '''
#k-means
km = KMeans(n_clusters=bin_count, n_init=50, random_state=0)
km_predictions = km.fit_predict(X)

#db scan
db = DBSCAN(eps=1.6, min_samples=5)
db_predictions = db.fit_predict(X)








print("BINS")
print(bins)
print()
print("km_predictions")
print(km_predictions)
print()
print("db_predictions")
print(db_predictions)
print(len(bins) == len(km_predictions) == len(db_predictions) == meal_matrix.shape[0])
print(bins.min() == 0, bins.max() == 5)
print(set(db_predictions))
# min/max carb 
# split into n bins - bin sz 20 grams, create bins (mincarb to min+20, min+21 to min+40, .... to maxcarb)
# Px1 bin matrix  is ground truth

#implement 2 clustering algos