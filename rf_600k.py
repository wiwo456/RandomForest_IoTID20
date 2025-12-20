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
data = pd.read_csv("IoTID20.csv")

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

batch_size = 50000
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
