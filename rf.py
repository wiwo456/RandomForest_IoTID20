'''
This code is about runnign the 36 features that were run in the reseach paper. Here, F1 score, accuracy, precison all is calculated.
this code has river library for adaptive random forest and extremely fast decision tree, the run time for this code was around 1 hr.
this has a batch separation of 10000 and performs 9 cycles uptop 100000 rows with 36 features in it.
'''
'''

This is the ouput that i got for this run.

Training Static Random Forest.

STATIC RANDOM FOREST RESULTS (36 FEATURES)
Accuracy : 98.73%
Precision: 98.71%
Recall   : 98.73%
F1-Score : 98.70%

Initializing Adaptive Model.

Starting streaming in 9 batches of 10000 samples each...

Batch 1/9 | Accuracy=0.9684 | Precision=0.9663 | Recall=0.9684 | F1=0.9667
Batch 2/9 | Accuracy=0.9712 | Precision=0.9727 | Recall=0.9741 | F1=0.9727
Batch 3/9 | Accuracy=0.9713 | Precision=0.9699 | Recall=0.9713 | F1=0.9695
Batch 4/9 | Accuracy=0.9724 | Precision=0.9750 | Recall=0.9759 | F1=0.9744
Batch 5/9 | Accuracy=0.9732 | Precision=0.9756 | Recall=0.9764 | F1=0.9749
Batch 6/9 | Accuracy=0.9744 | Precision=0.9795 | Recall=0.9801 | F1=0.9790
Batch 7/9 | Accuracy=0.9746 | Precision=0.9752 | Recall=0.9762 | F1=0.9750
Batch 8/9 | Accuracy=0.9754 | Precision=0.9807 | Recall=0.9811 | F1=0.9802
Batch 9/9 | Accuracy=0.9759 | Precision=0.9791 | Recall=0.9798 | F1=0.9788

ADAPTIVE RIVER MODEL (EFD TREE) - MEAN METRICS (10 BATCHES)
Mean Accuracy : 0.9730
Mean Precision: 0.9749
Mean Recall   : 0.9759
Mean F1-Score : 0.9746

Adaptive random forest complete
'''
import time 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from river import ensemble, tree, metrics

start_time = time.time()
print("Loading IoTID20 dataset...")
#data = pd.read_csv("IoTID20.csv", nrows=100000)
data = pd.read_csv("/content/drive/MyDrive/datasets/IoTID20.csv", nrows=100000)
print("Dataset loaded! Shape:", data.shape)

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
print(f"Number of selected features found: {len(features)}")


X = data[features]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("\nTraining Static Random Forest.")
rf = RandomForestClassifier(random_state=0, n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


print("STATIC RANDOM FOREST RESULTS (36 FEATURES)")

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")



print("\nInitializing Adaptive Model.")

river_model = ensemble.LeveragingBaggingClassifier(
    model=tree.ExtremelyFastDecisionTreeClassifier(grace_period=50),
    n_models=5,
    seed=42
)
acc_metric = metrics.Accuracy()

X_dict = X.to_dict(orient='records')
y_values = y.values

batch_size = 10000
num_batches = len(X) // batch_size

batch_acc, batch_prec, batch_rec, batch_f1 = [], [], [], []

print(f"\nStarting streaming in {num_batches} batches of {batch_size} samples each...\n")

for i in range(num_batches):
    start, end = i * batch_size, (i + 1) * batch_size
    X_batch, y_batch = X_dict[start:end], y_values[start:end]

    y_true, y_pred = [], []

    for xi, yi in zip(X_batch, y_batch):
        y_hat = river_model.predict_one(xi)
        if y_hat is not None:
            y_true.append(yi)
            y_pred.append(y_hat)
            acc_metric.update(yi, y_hat)
        river_model.learn_one(xi, yi)

    if len(y_true) > 0:
        precision_b = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_b = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_b = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        acc_b = acc_metric.get()

        batch_acc.append(acc_b)
        batch_prec.append(precision_b)
        batch_rec.append(recall_b)
        batch_f1.append(f1_b)

        print(f"Batch {i+1}/{num_batches} | "
              f"Accuracy={acc_b:.4f} | Precision={precision_b:.4f} | "
              f"Recall={recall_b:.4f} | F1={f1_b:.4f}")


print(" RIVER MODEL (EFD TREE) - MEAN METRICS (10 BATCHES)")

print(f"Mean Accuracy : {np.mean(batch_acc):.4f}")
print(f"Mean Precision: {np.mean(batch_prec):.4f}")
print(f"Mean Recall   : {np.mean(batch_rec):.4f}")
print(f"Mean F1-Score : {np.mean(batch_f1):.4f}")

print("Adaptive random forest complete")

print(f"\nTotal runtime: {(time.time() - start_time)/60:.3f} minutes")