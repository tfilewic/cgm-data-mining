#tfilewic  2025-09-25
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

CGM_DATA_PATH = "CGMData.csv"
INSULIN_DATA_PATH = "InsulinData.csv"
RESULT_PATH = "Result.csv"

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


def create_feature_row(glucose: np.ndarray) -> np.ndarray:
    """
    creates a row of features from a period of glucose readings
    """
    n = len(glucose)
    minimum = float(glucose.min())
    maximum = float(glucose.max())

    quarter = n // 4
    quarter_slope = (glucose[quarter+1] - glucose[quarter-1]) / 2.0

    smoothed3 = np.convolve(glucose, np.array([1,1,1])/3.0, mode="same")
    start = n // 5  #ignore first 20%
    end = n - 2     #ignore trailing edge
    peak_index = start + int(np.argmax(smoothed3[start:end]))
    ttp = peak_index / (n - 1)

    normalized_difference = (maximum - minimum) / minimum
    range = maximum - minimum
    
    d1_3 = glucose[2:] - glucose[:-2]  #across 3pts
    max_d1_3 = np.max(np.abs(d1_3))

    d2 = np.diff(np.diff(glucose))
    max_d2 = np.max(np.abs(d2))
    
    fft_vals = np.fft.fft(glucose)
    power = np.abs(fft_vals) ** 2
    fft = power[1]

    return np.array(
        [ttp, normalized_difference, fft, range, max_d1_3, max_d2, quarter_slope],
        dtype=float
    )

def extract_features(matrix: np.ndarray) -> np.ndarray:
    """
    extracts features from each row in a meal or nomeal matrix
    """
    features = [create_feature_row(row) for row in matrix]
    return np.vstack(features)



def calculate_metrics(ground_truth_bins: np.ndarray, bin_count: int, predicted_labels: np.ndarray, scaled_meal_matrix: np.ndarray):
    """
    calculates sse, entropy, and purity metrics from clustering results
    """
    sse = 0.0
    entropy = 0.0
    purity = 0.0
    
    mask = predicted_labels != -1   #drop dbscan -1s

    y = ground_truth_bins[mask]
    X = scaled_meal_matrix[mask]
    labels = predicted_labels[mask]
    N = len(y)

    for cluster_id in np.unique(labels):
        indices = (labels == cluster_id)
        cluster_size = indices.sum()
        counts = np.bincount(y[indices], minlength=bin_count)
        probabilities = counts / cluster_size
        nonzero_probabilities = probabilities[probabilities > 0]
        entropy += -(cluster_size / N) * np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))
        centroid = X[indices].mean(axis=0)

        diff = X[indices] - centroid
        sse += (diff * diff).sum()
        purity += (cluster_size / N) * probabilities.max()

    return sse, entropy, purity




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

''' EXTRACT AND SCALE FEATURES '''
meal_features = extract_features(meal_matrix)

#X = StandardScaler().fit_transform(meal_features)
X = MinMaxScaler().fit_transform(meal_features)


''' CLUSTERING '''
#k-means
km = KMeans(n_clusters=bin_count, n_init=5, max_iter=18, random_state=0)
km_predictions = km.fit_predict(X)

#db scan
db = DBSCAN(eps=0.17, min_samples=5)
db_predictions = db.fit_predict(X)


''' CALCULATE METRICS '''
#KMeans
sse_km, ent_km, pur_km = calculate_metrics(bins, bin_count, km_predictions, X)

#DBSCAN
sse_db, ent_db, pur_db = calculate_metrics(bins, bin_count, db_predictions, X)

#export to csv
results = np.array([[sse_km, sse_db, ent_km, ent_db, pur_km, pur_db]])
print(results)
np.savetxt(RESULT_PATH, results, delimiter=",")

