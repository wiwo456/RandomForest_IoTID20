import time 
import pandas as pd
import numpy as np
from collections import deque

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from river import ensemble, tree, metrics
from river.drift import ADWIN


start_time = time.time()
print("Loading IoTID20 dataset...")

data = pd.read_csv("IoTID20.csv", nrows=100000)
print("Loaded shape:", data.shape)

data = data.drop_duplicates()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

remove_cols = ['Src_IP', 'Dst_IP', 'Flow_ID', 'Sub_Cat', 'Timestamp']
data = data.drop(columns=[c for c in remove_cols if c in data.columns], errors='ignore')

encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])

features = [
    'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts',
    'Fwd_Pkt_Len_Max', 'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Fwd_Pkt_Len_Std',
    'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Bwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Std',
    'Flow_Byts/s', 'Flow_Pkts/s', 'Flow_IAT_Mean', 'Flow_IAT_Std',
    'Fwd_IAT_Tot', 'Fwd_IAT_Mean', 'Bwd_IAT_Tot', 'Bwd_IAT_Mean',
    'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Fwd_PSH_Flags',
    'FIN_Flag_Cnt', 'SYN_Flag_Cnt', 'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt',
    'Pkt_Size_Avg', 'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts',
    'Active_Mean', 'Active_Std', 'Idle_Mean', 'Idle_Std'
]

features = [f for f in features if f in data.columns]
print("Using features:", len(features))

X = data[features]
y = data['Label']


print("\nTraining Static Random Forest...\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(random_state=0, n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nSTATIC RANDOM FOREST RESULTS")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1-Score :", f1_score(y_test, y_pred, average='weighted', zero_division=0))


print("\nInitializing Adaptive Model...\n")

river_model = ensemble.LeveragingBaggingClassifier(
    model=tree.ExtremelyFastDecisionTreeClassifier(grace_period=50),
    n_models=5,
    seed=42
)

acc_metric = metrics.Accuracy()
drift_detector = ADWIN() 

prequential_window = deque(maxlen=1000)   # prequential window of size 1000
batch_size = 500

X_dict = X.to_dict(orient='records')
y_values = y.values

num_batches = len(X) // batch_size

batch_acc, batch_prec, batch_rec, batch_f1 = [], [], [], []

print(f"Starting streaming in {num_batches} batches of {batch_size}...\n")

for i in range(num_batches):

    start, end = i * batch_size, (i + 1) * batch_size
    X_batch, y_batch = X_dict[start:end], y_values[start:end]

    y_true, y_pred = [], []

    for idx, (xi, yi) in enumerate(zip(X_batch, y_batch)):

        y_hat = river_model.predict_one(xi)

        if y_hat is not None:
            y_true.append(yi)
            y_pred.append(y_hat)
            acc_metric.update(yi, y_hat)

        river_model.learn_one(xi, yi)

        error = int(y_hat != yi if y_hat is not None else 0)
        prequential_window.append(error)

        drift_detector.update(error)

        if drift_detector.drift_detected:
            print(f"⚠ Drift detected at batch {i+1}, sample {idx+1}")

    if len(y_true) > 0:
        batch_acc.append(accuracy_score(y_true, y_pred))
        batch_prec.append(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        batch_rec.append(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        batch_f1.append(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        print(f"Batch {i+1}/{num_batches} | "
              f"Accuracy={batch_acc[-1]:.4f} | "
              f"Precision={batch_prec[-1]:.4f} | "
              f"Recall={batch_rec[-1]:.4f} | "
              f"F1={batch_f1[-1]:.4f}")


print("\nADAPTIVE MODEL (MEAN METRICS)")
print("Mean Accuracy :", np.mean(batch_acc))
print("Mean Precision:", np.mean(batch_prec))
print("Mean Recall   :", np.mean(batch_rec))
print("Mean F1-Score :", np.mean(batch_f1))

print("\nAdaptive model complete!")
print("\nTotal runtime:", (time.time() - start_time)/60, "minutes")
