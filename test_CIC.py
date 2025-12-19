import time
import pandas as pd
import numpy as np
from collections import deque

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from river import ensemble, tree, metrics
from river.drift import ADWIN


start_time = time.time()
print("Starting CIC IoT 2023 Static vs Streaming Comparison...")

file_path = "/Users/hussain/code/CIC_IoT2023_3GB.csv"
data = pd.read_csv(file_path, nrows=100000)
print(f"Loaded {len(data)} rows and {len(data.columns)} columns")

data = data.drop_duplicates()
data = data.dropna()
print(f"After cleaning: {len(data)} rows remaining")



print("\n[0] Attack vs Normal Ratio")

normal_keywords = ["normal", "benign", "benign_traffic"]

binary = data["label"].astype(str).str.lower().apply(
    lambda x: 0 if any(key in x for key in normal_keywords) else 1
)

ratio = binary.value_counts(normalize=True) * 100

attack_percent = ratio.get(1, 0)
normal_percent = ratio.get(0, 0)

print(f"Attack: {attack_percent:.2f}%")
print(f"Normal: {normal_percent:.2f}%")




encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["label"])

X = data.drop(columns=["label"])
y = data["label"]



print("\n[1] Training static Random Forest on 100k rows...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_static = rf.predict(X_test)

print("\nSTATIC MODEL RESULTS (Random Forest)")
print(f"Accuracy : {accuracy_score(y_test, y_pred_static) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_static, average='weighted', zero_division=0) * 100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred_static, average='weighted', zero_division=0) * 100:.2f}%")
print(f"F1-Score : {f1_score(y_test, y_pred_static, average='weighted', zero_division=0) * 100:.2f}%")


print("\n[2] Starting Streaming with EFDT + LeveragingBagging + ADWIN Drift...\n")

river_model = ensemble.LeveragingBaggingClassifier(
    model=tree.ExtremelyFastDecisionTreeClassifier(grace_period=50),
    n_models=5,
    seed=42
)

drift_detector = ADWIN(delta=0.05) #adwin 0.05

batch_size = 50
num_batches = len(X) // batch_size
print(f"Streaming in {num_batches} mini-batches of {batch_size} samples each...\n")

prequential_window = deque(maxlen=1000)

acc_list, prec_list, rec_list, f1_list = [], [], [], []
drift_points = []    

X_dict = X.to_dict(orient="records")
y_array = y.values

for batch in range(num_batches):

    start = batch * batch_size
    end = start + batch_size

    X_batch = X_dict[start:end]
    y_batch = y_array[start:end]

    y_true, y_pred = [], []

    for i, (xi, yi) in enumerate(zip(X_batch, y_batch)):

        y_hat = river_model.predict_one(xi)
        river_model.learn_one(xi, yi)

        if y_hat is not None:
            y_true.append(yi)
            y_pred.append(y_hat)

        error = int(y_hat != yi if y_hat is not None else 0)
        prequential_window.append(error)

        drift_detector.update(error)

        if drift_detector.drift_detected:
            print(f"⚠ Drift detected at batch {batch+1}, sample {i+1}")
            drift_points.append((batch+1, i+1))

    
    if len(y_pred) > 0:
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        print(
            f"Batch {batch + 1}/{num_batches} | "
            f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
        )



print("\nSTREAMING MODEL RESULTS (EFDT + LeveragingBagging)")
print(f"Mean Accuracy : {np.mean(acc_list) * 100:.2f}%")
print(f"Mean Precision: {np.mean(prec_list) * 100:.2f}%")
print(f"Mean Recall   : {np.mean(rec_list) * 100:.2f}%")
print(f"Mean F1-Score : {np.mean(f1_list) * 100:.2f}%")



print("\n================ DRIFT SUMMARY ================")
if len(drift_points) == 0:
    print("No drift detected in this run.")
else:
    print(f"Total drifts detected: {len(drift_points)}")
    for b, s in drift_points:
        print(f"- Drift at batch {b}, sample {s}")
print("===============================================\n")

print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")
