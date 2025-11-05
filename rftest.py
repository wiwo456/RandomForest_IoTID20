'''
This code is about runnign the 11 features that I found as some of the most important features in anomaly detection. Here, F1 score, accuracy, precison all is calculated.
this code has river library for adaptive random forest and extremely fast decision tree.
this has a batch separation of 10000 and performs 9 cycles uptop 100000 rows with 11 features in it.


This is the output for this 


Loading IoTID20 dataset...
Dataset loaded! Shape: (100000, 86)

Training Static Random Forest (baseline)...

STATIC RANDOM FOREST RESULTS (11 FEATURES)
Accuracy : 98.42%
Precision: 98.40%
Recall   : 98.42%
F1-Score : 98.36%

Initializing Adaptive Model (River - LeveragingBagging w/ EFD Tree)...

Starting streaming in 9 batches of 10000 samples each...

Batch 1/9 | Accuracy=94.79% | Precision=93.88% | Recall=94.79% | F1=93.87%
Batch 2/9 | Accuracy=95.10% | Precision=94.85% | Recall=95.42% | F1=94.83%
Batch 3/9 | Accuracy=95.55% | Precision=96.25% | Recall=96.45% | F1=96.10%
Batch 4/9 | Accuracy=95.82% | Precision=96.40% | Recall=96.63% | F1=96.38%
Batch 5/9 | Accuracy=96.11% | Precision=97.15% | Recall=97.26% | F1=97.05%
Batch 6/9 | Accuracy=96.32% | Precision=97.26% | Recall=97.38% | F1=97.20%
Batch 7/9 | Accuracy=96.45% | Precision=97.15% | Recall=97.25% | F1=97.03%
Batch 8/9 | Accuracy=96.60% | Precision=97.51% | Recall=97.59% | F1=97.44%
Batch 9/9 | Accuracy=96.75% | Precision=97.90% | Recall=97.95% | F1=97.82%

ADAPTIVE RIVER MODEL (EFD TREE) - MEAN METRICS
Mean Accuracy : 95.94%
Mean Precision: 96.48%
Mean Recall   : 96.75%
Mean F1-Score : 96.41%

Adaptive Random Forest complete.

Total runtime: 10.040 minutes
'''

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from river import ensemble, tree, metrics

start_time = time.time()

print("Loading IoTID20 dataset...")
data = pd.read_csv("IoTID20.csv", nrows=100000)
print("Dataset loaded! Shape:", data.shape)

data = data.drop_duplicates()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

remove_cols = ['Src_IP', 'Dst_IP', 'Flow_ID', 'Sub_Cat', 'Timestamp']
data = data.drop(columns=[c for c in remove_cols if c in data.columns], errors='ignore')

if 'Label' not in data.columns:
    raise ValueError("Label column not found!")

encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])

features = [
    'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts',
    'Flow_Byts/s', 'Flow_Pkts/s',
    'Pkt_Len_Mean', 'Pkt_Len_Std',
    'Flow_IAT_Mean', 'Flow_IAT_Std',
    'Active_Mean', 'Idle_Mean'
]
features = [f for f in features if f in data.columns]

X = data[features]
y = data['Label']

print("\nTraining Static Random Forest (baseline)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

y_test_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred) * 100
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0) * 100
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0) * 100
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0) * 100

print("\nSTATIC RANDOM FOREST RESULTS (11 FEATURES)")
print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-Score : {f1:.2f}%")

print("\nInitializing Adaptive Model (River - LeveragingBagging w/ EFD Tree)...")
river_model = ensemble.LeveragingBaggingClassifier(
    model=tree.ExtremelyFastDecisionTreeClassifier(grace_period=50),
    n_models=5,
    seed=42
)

X_dict = X.to_dict(orient='records')
y_values = y.values

batch_size = 10000
num_batches = len(X) // batch_size
acc_metric = metrics.Accuracy()

batch_acc, batch_prec, batch_rec, batch_f1 = [], [], [], []

print(f"\nStarting streaming in {num_batches} batches of {batch_size} samples each...\n")
for i in range(num_batches):
    start, end = i * batch_size, (i + 1) * batch_size
    X_batch, y_batch = X_dict[start:end], y_values[start:end]

    y_true, y_pred = [], []
    for xi, yi in zip(X_batch, y_batch):
        y_hat = river_model.predict_one(xi)
        if y_hat is not None:
            acc_metric.update(yi, y_hat)
            y_true.append(yi)
            y_pred.append(y_hat)
        river_model.learn_one(xi, yi)

    if len(y_true) > 0:
        batch_acc.append(acc_metric.get() * 100)
        batch_prec.append(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100)
        batch_rec.append(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100)
        batch_f1.append(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100)
        print(f"Batch {i+1}/{num_batches} | Accuracy={batch_acc[-1]:.2f}% | Precision={batch_prec[-1]:.2f}% | Recall={batch_rec[-1]:.2f}% | F1={batch_f1[-1]:.2f}%")

print("\nADAPTIVE RIVER MODEL (EFD TREE) - MEAN METRICS")
print(f"Mean Accuracy : {np.mean(batch_acc):.2f}%")
print(f"Mean Precision: {np.mean(batch_prec):.2f}%")
print(f"Mean Recall   : {np.mean(batch_rec):.2f}%")
print(f"Mean F1-Score : {np.mean(batch_f1):.2f}%")

print("\nAdaptive Random complete.")

print(f"\nTotal runtime: {(time.time() - start_time)/60:.3f} minutes")
